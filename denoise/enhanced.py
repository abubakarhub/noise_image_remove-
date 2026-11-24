"""Enhanced, noise-aware denoisers (beyond the four classic filters of the synopsis).

Each method takes its strength from the no-reference noise analysis, so no clean
reference and no manual tuning is needed:

adaptive_median  salt & pepper: only pixels detected as impulses are replaced, by the
                 median of the clean (non-impulse) pixels around them; the window grows
                 3x3 -> 5x5 -> ... until enough clean neighbours are found. All other
                 pixels keep their exact original value.
nlm              additive Gaussian: OpenCV non-local means (Buades et al., 2005) with
                 the filter strength h set from the estimated sigma.
vst_nlm          signal-dependent noise (Poisson, speckle): a variance-stabilising
                 transform T(v) = integral dv / sd(v), built from the fitted noise
                 curve sd(v)^2 = a + b v + c v^2, makes the noise roughly N(0, 1) at
                 every intensity; non-local means removes it; the inverse transform
                 restores the original intensity scale. For Poisson this is the
                 Anscombe transform, for speckle a logarithm.
auto             detect -> estimate -> run the matching method above.
"""
import cv2
import numpy as np

from .analysis import detect_noise

ENHANCED_NAMES = {
    "auto": "Auto",
    "adaptive_median": "Adaptive median",
    "nlm": "NL-means",
    "vst_nlm": "VST + NLM",
}

# Filter strength h = factor x noise sd. The best factor grows with how heavy the noise is
# (heavy noise is less Gaussian after clipping), so it is read from tables tuned on 15
# held-out images (experiments/eval_denoise.py reports the result on the project samples).
# Severity: sigma for Gaussian; relative noise sd at mid-grey for Poisson/speckle.
_H_TABLE = {
    "gaussian": ([10, 15, 25, 40, 60], [1.0, 1.0, 0.85, 0.85, 0.95]),
    "poisson": ([0.13, 0.32, 0.5, 0.7], [1.0, 0.9, 1.3, 1.7]),
    "speckle": ([0.14, 0.28, 0.4, 0.55], [1.0, 1.0, 2.2, 4.6]),
}
# Colour images are denoised channel by channel: the synthetic noise is independent per
# channel, and this beat OpenCV's Lab-space colour NLM by +0.03..0.07 SSIM on every noise type.


def h_factor(kind, severity):
    xs, ys = _H_TABLE[kind]
    return float(np.interp(severity, xs, ys))


def _per_channel(fn, img):
    if img.ndim == 2:
        return fn(img)
    return cv2.merge([fn(np.ascontiguousarray(c)) for c in cv2.split(img)])


def severity(kind, k):
    """sigma (Gaussian) or relative noise sd at mid-grey (Poisson, speckle)."""
    if kind == "gaussian":
        return float(np.sqrt(k))
    if kind == "poisson":
        return float(np.sqrt(k * 128.0) / 128.0)
    return float(np.sqrt(k))


def _odd(k):
    k = max(3, int(k))
    return k if k % 2 else k + 1


# ---------------------------------------------------------------- salt & pepper
def _impulses(ch):
    med = cv2.medianBlur(ch, 5)
    return ((ch == 0) | (ch == 255)) & (cv2.absdiff(ch, med) > 40)


def _adaptive_median_channel(ch, max_k=15, min_good=3):
    bad = _impulses(ch)
    if not bad.any():
        return ch.copy()
    out = ch.copy()
    todo = bad.copy()
    good = (~bad).astype(np.float32)
    val = np.where(bad, 0, ch).astype(np.float32)
    h, w = ch.shape
    for k in range(3, max_k + 1, 2):
        ys, xs = np.nonzero(todo)
        if ys.size == 0:
            break
        r = k // 2
        cnt = cv2.boxFilter(good, -1, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
        ready = cnt[ys, xs] >= min_good
        if k == max_k:
            ready[:] = True
        ys, xs = ys[ready], xs[ready]
        if ys.size == 0:
            continue
        # median of the clean neighbours (impulses masked out as NaN)
        pad = cv2.copyMakeBorder(np.where(bad, np.nan, val).astype(np.float32), r, r, r, r, cv2.BORDER_REFLECT)
        win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))[ys, xs].reshape(ys.size, -1)
        with np.errstate(all="ignore"):
            med = np.nanmedian(win, axis=1)
        fallback = np.isnan(med)
        if fallback.any():
            med[fallback] = cv2.medianBlur(ch, 5)[ys[fallback], xs[fallback]]
        out[ys, xs] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
        todo[ys, xs] = False
    return out


def adaptive_median(img):
    if img.ndim == 2:
        return _adaptive_median_channel(img)
    return cv2.merge([_adaptive_median_channel(np.ascontiguousarray(c)) for c in cv2.split(img)])


# ---------------------------------------------------------------- Gaussian
def nlm(img, sigma, factor=None):
    """Non-local means with h derived from the noise standard deviation."""
    sigma = max(float(sigma), 0.5)
    h = (factor or h_factor("gaussian", sigma)) * sigma
    return _per_channel(lambda ch: cv2.fastNlMeansDenoising(ch, None, h, 7, 21), img)


# ---------------------------------------------------------------- Poisson / speckle
def _vst_tables(model, floor=0.25):
    """Forward/inverse lookup for T(v) = integral_0^v du / sqrt(a + b u + c u^2 + floor)."""
    a, b, c = model
    v = np.linspace(0.0, 255.0, 4097)
    sd = np.sqrt(np.maximum(a + b * v + c * v * v, 0.0) + floor)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (1 / sd[1:] + 1 / sd[:-1]) * np.diff(v))])
    return v, t


_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(48)  # nodes/weights for E[g(Z)], Z ~ N(0,1)
_GH_W = _GH_W / _GH_W.sum()


def _expected_transform(kind, k, v, t):
    """E[T(Y_f)] for f = 0..255, where Y_f is the clipped noisy value under the model.

    Inverting this (instead of T itself) is the exact unbiased inverse of Makitalo & Foi
    (2011), generalised here to clipping and to speckle; it removes the darkening that a
    plain algebraic inverse causes after averaging in the transformed domain.
    """
    f = np.arange(256, dtype=np.float64)
    if kind == "poisson" and 255.0 / k <= 150:
        from math import lgamma
        peak = 255.0 / k
        cm = int(np.ceil(peak)) + 1
        c = np.arange(cm, dtype=np.float64)
        lam = (f * peak / 255.0)[:, None]
        lg = np.array([lgamma(x + 1) for x in c])
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(lam > 0, c * np.log(np.maximum(lam, 1e-300)) - lam - lg, np.where(c == 0, 0.0, -np.inf))
        pmf = np.exp(logp)
        tail = np.clip(1.0 - pmf.sum(axis=1), 0.0, 1.0)
        y = np.minimum(np.rint(c * 255.0 / peak), 255.0)
        return (pmf * np.interp(y, v, t)[None, :]).sum(axis=1) + tail * t[-1]
    sd = {"gaussian": np.full_like(f, np.sqrt(k)), "speckle": np.sqrt(k) * f,
          "poisson": np.sqrt(k * f)}[kind]
    y = np.clip(np.rint(f[:, None] + sd[:, None] * _GH_Z[None, :]), 0, 255)
    return (np.interp(y, v, t) * _GH_W[None, :]).sum(axis=1)


def vst_nlm(img, kind, k, factor=None):
    """Stabilise variance with the fitted noise curve, denoise with NLM, invert without bias."""
    model = {"gaussian": (k, 0, 0), "poisson": (0, k, 0), "speckle": (0, 0, k)}[kind]
    v, t = _vst_tables(model)
    span = float(t[-1])
    if span <= 0:
        return img.copy()
    scale = 255.0 / span                      # transformed noise sd is ~1 -> `scale` grey levels
    fwd = np.interp(img.astype(np.float64), v, t) * scale
    tr = np.clip(np.rint(fwd), 0, 255).astype(np.uint8)
    h = (factor or h_factor(kind, severity(kind, k))) * scale
    den = _per_channel(lambda ch: cv2.fastNlMeansDenoising(ch, None, h, 7, 21), tr)
    et = np.maximum.accumulate(_expected_transform(kind, k, v, t))  # monotone for interp
    back = np.interp(den.astype(np.float64) / scale, et, np.arange(256, dtype=np.float64))
    return np.clip(np.rint(back), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- dispatcher
def _model_for(info):
    """Variance model (a, b, c) from the analysis; Gaussian fallback from the sigma read-out."""
    a, b, c = info.get("model", (0.0, 0.0, 0.0))
    if a + b + c == 0:
        s = info.get("mid_sd") or info.get("sigma", 5.0)
        a = float(s) ** 2
    return a, b, c


def apply_enhanced(img, method, info=None):
    """Run one enhanced method. `info` is a detect_noise() result (computed if omitted)."""
    if method == "adaptive_median":
        return adaptive_median(img), info
    info = info or detect_noise(img)
    if method == "auto":
        kind = info["type"]
        if kind == "none":
            return img.copy(), info
        method = info["enhanced"]["filter"]
        if method == "adaptive_median":
            return adaptive_median(img), info
    a, b, c = _model_for(info)
    if method == "nlm":
        # NLM needs one sd: use the model's value at mid-grey
        sd = np.sqrt(a + b * 128.0 + c * 128.0 ** 2)
        return nlm(img, sd), info
    if method == "vst_nlm":
        kind = info["type"] if info["type"] in ("poisson", "speckle") else ("speckle" if c > 0 else "poisson" if b > 0 else "gaussian")
        k = {"gaussian": a, "poisson": b, "speckle": c}[kind]
        return vst_nlm(img, kind, k), info
    raise ValueError(f"Unknown enhanced method: {method}")

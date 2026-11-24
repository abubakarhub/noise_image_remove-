"""No-reference noise analysis: detect the noise type and estimate its strength.

Objective 1 of the synopsis asks to "detect and analyse the noise present in a
given image". This works without the clean original, in two stages:

1. Impulse test - salt-and-pepper pixels are isolated extremes (0 or 255) that
   disagree strongly with their 5x5 median, while the *other* pixels stay quiet.
   (Heavy Poisson or clipped Gaussian noise also produces 0/255 pixels, but then
   the remaining pixels are noisy too, which separates the two cases.)

2. Noise-level function - the image is split into intensity bins using a 5x5
   median as the estimate of the clean signal. In each bin the noise standard
   deviation is measured on the flattest pixels from one quartile of the
   residual (the quartile on the side away from 0/255), so clipping at black or
   white does not bias it. Three single-term models are then fitted to the
   variance-vs-intensity points:
       var = a          additive Gaussian   (signal independent)
       var = b * f      Poisson             (variance proportional to intensity)
       var = c * f^2    speckle             (std. deviation proportional to intensity)
   The model with the smallest log-error wins and its parameter gives the noise
   strength (sigma, photon peak = 255/b, or speckle variance = c), which the
   enhanced denoisers use to tune themselves.
"""
import cv2
import numpy as np

# Best classic filter by noise type, as measured by experiments/run_experiments.py
# (Table 2 of the synopsis predicted median for speckle; Gaussian won in practice)
RECOMMENDED = {
    "salt_pepper": {"filter": "median", "ksize": 3,
                    "why": "Order statistics reject extreme outliers instead of averaging them into the neighbourhood."},
    "gaussian": {"filter": "bilateral", "ksize": 7,
                 "why": "Weighted local averaging reduces zero-mean additive noise; the range kernel keeps edges."},
    "speckle": {"filter": "gaussian", "ksize": 7,
                "why": "Measured best in our runs: Gaussian weighting beat median on granular multiplicative noise."},
    "poisson": {"filter": "gaussian", "ksize": 7,
                "why": "Smooth weighting suits signal-dependent shot noise, especially in dark regions."},
    "none": {"filter": "bilateral", "ksize": 5,
             "why": "Very little noise detected - a light edge-preserving pass is enough."},
}

# Enhanced (noise-aware) method chosen by the Auto mode for each detected type
ENHANCED = {
    "salt_pepper": {"filter": "adaptive_median",
                    "why": "Only the detected impulse pixels are replaced, from their clean neighbours; every other pixel is left untouched."},
    "gaussian": {"filter": "nlm",
                 "why": "Non-local means averages similar patches from the whole neighbourhood, with strength set from the estimated sigma."},
    "speckle": {"filter": "vst_nlm",
                "why": "A log-type variance-stabilising transform turns multiplicative noise into additive noise, which non-local means then removes."},
    "poisson": {"filter": "vst_nlm",
                "why": "The Anscombe-type transform makes shot noise uniform across intensities before non-local means; the inverse restores brightness."},
    "none": {"filter": "none",
             "why": "No significant noise detected - the image is returned unchanged to avoid needless blurring."},
}

_LAPLACE_DIFF = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_Q75 = 0.6744898  # 75th percentile of the standard normal
NONE_SIGMA = 1.6  # quietest 30% of the image below this ...
NONE_MID = 3.0    # ... or model-predicted noise sd at mid-grey below this -> clean


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def global_sigma(img):
    """Immerkaer (1996) fast noise standard-deviation estimate (kept for the UI read-out)."""
    g = _gray(img).astype(np.float64)
    h, w = g.shape
    conv = cv2.filter2D(g, -1, _LAPLACE_DIFF)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


# ---------------------------------------------------------------- stage 1
_RING = np.ones((3, 3), np.uint8)
_RING[1, 1] = 0


def spike_mask(g, thresh=40):
    """Isolated spikes: pixels that differ by more than `thresh` from ALL 8 neighbours."""
    gi = g.astype(np.int16)
    hi = gi - cv2.dilate(g, _RING).astype(np.int16) > thresh
    lo = cv2.erode(g, _RING).astype(np.int16) - gi > thresh
    return hi | lo


def impulse_mask(g, thresh=40):
    """Candidate salt/pepper pixels: 0/255 values that disagree with their 5x5 median."""
    extreme = (g == 0) | (g == 255)
    return extreme & (cv2.absdiff(g, cv2.medianBlur(g, 5)) > thresh)


def _impulse_stats(g):
    """Share of isolated 0/255 spikes, other spikes per 0/255 spike, and loudness of the rest."""
    spikes = spike_mask(g)
    extreme = (g == 0) | (g == 255)
    ext = float((spikes & extreme).mean())
    other = float((spikes & ~extreme).mean())
    rest = ~extreme
    d = cv2.absdiff(g, cv2.medianBlur(g, 5))
    loud = float((d[rest] > 60).mean()) if rest.any() else 0.0
    return ext, other / max(ext, 1e-9), loud


# ---------------------------------------------------------------- stage 2
def _block_means(a, block):
    h, w = a.shape
    hb, wb = h // block, w // block
    return a[: hb * block, : wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3)).ravel()


KAPPA = 0.73  # 20th-percentile / mean of the 8x8 block variance for pure Gaussian noise (calibrated)


def noise_level_function(g, block=8, bin_width=16, min_blocks=6, pct=20, kappa=None):
    """Observed noise curve of the image.

    Returns (pts, white):
      pts   [(mean, variance, n_blocks)] per intensity bin, as observed (i.e. after any
            clipping at 0/255). Local variance = (Immerkaer Laplacian-difference
            response)^2 / 36 averaged over 8x8 blocks; within each bin the 20th
            percentile keeps flat blocks and rejects edges/texture.
      white (variance, n) of the *unclipped* noise at intensity 255, measured in large
            saturated white areas (paper, sky) from the lower quartile, or None.
    """
    gf = g.astype(np.float64)
    var_map = cv2.filter2D(gf, -1, _LAPLACE_DIFF) ** 2 / 36.0
    bm, bv = _block_means(gf, block), _block_means(var_map, block)
    pts = []
    for lo in range(0, 256, bin_width):
        sel = (bm >= lo) & (bm < lo + bin_width)
        if sel.sum() < min_blocks:
            continue
        bvs, bms = bv[sel], bm[sel]
        lo_v, v_p, hi_v = np.percentile(bvs, [max(pct - 10, 0), pct, pct + 10])
        near = (bvs >= lo_v) & (bvs <= hi_v)
        # intensity of the blocks the percentile actually came from (not the whole bin)
        mu = float(bms[near].mean()) if near.any() else float(bms.mean())
        pts.append((mu, float(max(v_p / (kappa or KAPPA), 0.09)), int(sel.sum())))

    # The white region is chosen on one pixel lattice and measured on the interleaved
    # one, so the selection cannot favour pixels whose own noise happened to be positive.
    white = None
    sel_grid, meas_grid = g[0::2, 0::2], g[1::2, 1::2]
    h, w = min(sel_grid.shape[0], meas_grid.shape[0]), min(sel_grid.shape[1], meas_grid.shape[1])
    if h > 8 and w > 8:
        med = cv2.medianBlur(np.ascontiguousarray(sel_grid[:h, :w]), 5)
        mask = cv2.erode((med == 255).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mask.sum() > 500:
            vals = meas_grid[:h, :w][mask].astype(np.float64)
            r = vals - 255.0 + np.random.default_rng(1).uniform(-0.5, 0.5, vals.size)
            if 0.2 < float((r < -0.5).mean()) < 0.8:
                sd = max(-float(np.percentile(r, 25)) / _Q75, 0.3)
                white = (sd * sd, 4 * int(mask.sum()) // (block * block))
    return pts, white


# ---------------------------------------------------------------- forward noise models
FLOOR = 1.0  # variance every bin carries anyway (8-bit rounding, faint texture)
_F = np.arange(256, dtype=np.float64)
_GRIDS = {  # candidate strengths k for var = k * f^p
    "gaussian": np.geomspace(0.25, 1.2e4, 121),      # sigma^2   (sigma 0.5 .. 110)
    "poisson": 255.0 / np.geomspace(1.0, 5000, 121),  # 255/peak  (peak 1 .. 5000)
    "speckle": np.geomspace(1e-4, 3.0, 121),          # variance of n
}
_P = {"gaussian": 0, "poisson": 1, "speckle": 2}


def _erf(x):
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorised
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def _censored_normal(mu, sd):
    """Mean and variance of clip(N(mu, sd^2), 0, 255)."""
    sd = np.maximum(sd, 1e-6)
    a, b = (0.0 - mu) / sd, (255.0 - mu) / sd
    pa, pb = np.exp(-a * a / 2) / np.sqrt(2 * np.pi), np.exp(-b * b / 2) / np.sqrt(2 * np.pi)
    ca, cb = 0.5 * (1 + _erf(a / np.sqrt(2))), 0.5 * (1 + _erf(b / np.sqrt(2)))
    mid = cb - ca
    m1 = 255.0 * (1 - cb) + mu * mid + sd * (pa - pb)
    m2 = 255.0 ** 2 * (1 - cb) + (mu * mu + sd * sd) * mid + 2 * mu * sd * (pa - pb) + sd * sd * (a * pa - b * pb)
    return m1, np.maximum(m2 - m1 * m1, 0.0)


def _poisson_curve(peak):
    """Mean and variance of min(round(C * 255/peak), 255), C ~ Poisson(f * peak/255)."""
    if peak > 150:  # normal approximation is accurate here
        return _censored_normal(_F, np.sqrt(255.0 * _F / peak))
    from math import lgamma
    cm = int(np.ceil(peak)) + 1
    c = np.arange(cm, dtype=np.float64)
    lam = (_F * peak / 255.0)[:, None]
    lg = np.array([lgamma(x + 1) for x in c])
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(lam > 0, c * np.log(np.maximum(lam, 1e-300)) - lam - lg, np.where(c == 0, 0.0, -np.inf))
    pmf = np.exp(logp)
    tail = np.clip(1.0 - pmf.sum(axis=1), 0.0, 1.0)
    y = np.minimum(np.rint(c * 255.0 / peak), 255.0)
    m1 = (pmf * y).sum(axis=1) + tail * 255.0
    m2 = (pmf * y * y).sum(axis=1) + tail * 255.0 ** 2
    return m1, np.maximum(m2 - m1 * m1, 0.0)


_CURVES = {}


def _curves(model):
    """Observed (mean, variance) over f = 0..255 for every candidate strength (cached)."""
    if model not in _CURVES:
        out = []
        for k in _GRIDS[model]:
            if model == "gaussian":
                out.append(_censored_normal(_F, np.full_like(_F, np.sqrt(k))))
            elif model == "speckle":
                out.append(_censored_normal(_F, np.sqrt(k) * _F))
            else:
                out.append(_poisson_curve(255.0 / k))
        _CURVES[model] = out
    return _CURVES[model]


def _fit_models(sets, floor=None):
    """Fit each noise model (with clipping) to the observed curve(s).

    sets: [(pts, white), ...] - one per colour channel; the noise strength is shared.
    Returns ({model: (weighted squared log-error, k)}, log-intensity spread, free slope).
    """
    floor = FLOOR if floor is None else floor
    n_obs = sum(len(p) + (wh is not None) for p, wh in sets)
    if n_obs < 2:
        return None
    errs = {m: np.zeros(len(_GRIDS[m])) for m in _P}
    tot = 0.0
    xs_all, mus, lvs = [], [], []
    for pts, white in sets:
        mu = np.array([p[0] for p in pts])
        lv = np.log(np.array([p[1] for p in pts])) if pts else np.empty(0)
        w = np.sqrt(np.array([p[2] for p in pts], dtype=np.float64)) if pts else np.empty(0)
        tot += w.sum()
        if white is not None:
            ww = np.sqrt(white[1])
            tot += ww
        for model, p in _P.items():
            for i, (k, (m_obs, v_obs)) in enumerate(zip(_GRIDS[model], _curves(model))):
                if len(mu):
                    errs[model][i] += float(np.sum(w * (lv - np.log(floor + np.interp(mu, m_obs, v_obs))) ** 2))
                if white is not None:
                    errs[model][i] += float(ww * (np.log(white[0]) - np.log(floor + k * 255.0 ** p)) ** 2)
        xs_all += list(mu) + ([255.0] if white is not None else [])
        mus += list(mu)
        lvs += list(lv)
    out = {}
    for model in _P:
        i = int(np.argmin(errs[model]))
        out[model] = (float(errs[model][i] / max(tot, 1e-9)), float(_GRIDS[model][i]))
    xs = np.log(np.maximum(np.array(xs_all), 1.0))
    spread = float(np.ptp(xs)) if xs.size else 0.0
    mus, lvs = np.array(mus), np.array(lvs)
    sel = mus >= 4
    slope = (float(np.polyfit(np.log(mus[sel]), lvs[sel], 1)[0])
             if sel.sum() >= 2 and np.ptp(np.log(mus[sel])) > 0.05 else None)
    return out, spread, slope


def noise_params(kind, k):
    """Human-readable strength + the variance model (a, b, c) of var = a + b f + c f^2."""
    if kind == "gaussian":
        return {"sigma": round(float(np.sqrt(k)), 2)}, (k, 0.0, 0.0)
    if kind == "poisson":
        return {"peak": round(float(255.0 / k), 2)}, (0.0, k, 0.0)
    if kind == "speckle":
        return {"variance": round(float(k), 4)}, (0.0, 0.0, k)
    return {}, (0.0, 0.0, 0.0)


def robust_sigma(g):
    """Median-based Immerkaer sigma over unclipped pixels (fallback when no flat blocks exist)."""
    conv = cv2.filter2D(g.astype(np.float64), -1, _LAPLACE_DIFF)
    clip = cv2.dilate(((g == 0) | (g == 255)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vals = np.abs(conv[~clip]) if (~clip).sum() > 1000 else np.abs(conv).ravel()
    return float(np.median(vals) / (_Q75 * 6.0))


def detect_noise(img):
    g = _gray(img)
    sigma = global_sigma(img)
    ext_spikes, spike_ratio, loud = _impulse_stats(g)
    channels = cv2.split(img) if img.ndim == 3 else [g]
    sets = [noise_level_function(np.ascontiguousarray(c)) for c in channels]
    fit = _fit_models(sets)
    # pooled curve for the clean-image test and the chart
    pts = sorted((p for ps, _ in sets for p in ps), key=lambda t: t[0])
    slope = None

    scores = {"salt_pepper": 0.0, "gaussian": 0.0, "speckle": 0.0, "poisson": 0.0}
    params, model = {}, (0.0, 0.0, 0.0)
    # typical noise std. deviation, weighted by how much of the image each bin covers
    if pts:
        sds = np.sqrt([v for _, v, _ in pts])
        wts = np.array([n for _, _, n in pts], dtype=np.float64)
        level = float(np.sum(sds * wts) / wts.sum())
        # clean-image test: the quietest 30% of the image (texture never reaches every bin)
        order = np.argsort(sds)
        cum = np.cumsum(wts[order]) / wts.sum()
        quiet = float(sds[order][np.searchsorted(cum, 0.3)])
    else:
        level = quiet = robust_sigma(g)
    # noise std. deviation the best-fitting model predicts at mid-grey (f = 128)
    mid_sd = level
    if fit is not None:
        m_best = min(fit[0], key=lambda m: fit[0][m][0])
        mid_sd = float(np.sqrt(fit[0][m_best][1] * 128.0 ** _P[m_best]))

    if ext_spikes > 0.002 and spike_ratio < 0.35 and loud < 0.33:
        kind = "salt_pepper"
        conf = float(min(0.99, 0.8 + 0.4 * (0.35 - spike_ratio) + 2 * ext_spikes))
        scores["salt_pepper"] = conf
        dens = float(impulse_mask(g).mean())
        params = {"density": round(min(1.0, dens), 4)}
    elif mid_sd < NONE_MID or (quiet < NONE_SIGMA and mid_sd < 2 * NONE_MID):
        kind = "none"
        conf = float(min(0.99, 0.6 + 0.4 * max(0.0, NONE_MID - mid_sd) / NONE_MID + (0.1 if quiet < NONE_SIGMA else 0)))
    elif fit is None or fit[1] < 0.35:
        # one intensity level only (e.g. a bi-level page): the three models predict the
        # same curve there, so only the strength can be measured reliably
        kind, conf = "gaussian", 0.4
        scores.update({"gaussian": 0.4, "poisson": 0.3, "speckle": 0.3})
        k = fit[0]["gaussian"][1] if fit else level ** 2
        params, model = noise_params("gaussian", k)
    else:
        errs, spread, slope = fit
        tau = 0.02 + 0.05 / spread
        best = min(e for e, _ in errs.values())
        raw = {k_: np.exp(-(e - best) / tau) for k_, (e, _) in errs.items()}
        tot = sum(raw.values())
        scores.update({k_: float(v / tot) for k_, v in raw.items()})
        kind = max(raw, key=raw.get)
        conf = scores[kind]
        params, model = noise_params(kind, errs[kind][1])

    return {
        "type": kind,
        "confidence": round(float(conf), 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "sigma": round(sigma, 2),
        "noise_level": round(level, 2),
        "mid_sd": round(mid_sd, 2),
        "impulse_ratio": round(ext_spikes, 4),
        "slope": None if slope is None else round(slope, 2),
        "params": params,
        "model": [float(x) for x in model],
        "variance_curve": [[round(m, 1), round(v, 2)] for m, v, _ in pts],
        "recommendation": RECOMMENDED[kind],
        "enhanced": ENHANCED[kind],
    }


def histogram(img, bins=64):
    g = _gray(img)
    h = cv2.calcHist([g], [0], None, [bins], [0, 256]).ravel()
    return (h / h.sum()).round(5).tolist()

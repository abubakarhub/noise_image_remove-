"""No-reference noise analysis: detect the noise type present in an image.

Objective 1 of the synopsis asks to "detect and analyse the noise present in a
given image". This works without the clean original:

1. Impulse test - salt-and-pepper pixels are isolated extremes (0 or 255) that
   disagree strongly with their 3x3 median.
2. Signal-dependence test - local noise variance is estimated per block with
   Immerkaer's Laplacian-difference operator and regressed against local mean
   intensity on a log-log scale. The slope b of  log(var) = a + b*log(mean)
   separates the remaining models:
       b ~ 0  -> additive Gaussian (signal independent)
       b ~ 1  -> Poisson (variance equals mean)
       b ~ 2  -> speckle (std. deviation proportional to intensity)
"""
import cv2
import numpy as np

# Best filter by noise type, as measured by experiments/run_experiments.py
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

_LAPLACE_DIFF = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def global_sigma(img):
    """Immerkaer (1996) fast noise standard-deviation estimate."""
    g = _gray(img).astype(np.float64)
    h, w = g.shape
    conv = cv2.filter2D(g, -1, _LAPLACE_DIFF)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def _impulse_stats(g):
    med = cv2.medianBlur(g, 3)
    deviant = np.abs(g.astype(np.int16) - med.astype(np.int16)) > 60
    extreme = (g == 0) | (g == 255)
    n_dev = int(deviant.sum())
    ratio = float((deviant & extreme).sum()) / g.size
    share = float((deviant & extreme).sum()) / n_dev if n_dev else 0.0
    return ratio, share


def _variance_slope(g, block=8):
    g = g.astype(np.float64)
    var_map = cv2.filter2D(g, -1, _LAPLACE_DIFF) ** 2 / 36.0
    mean_map = cv2.blur(g, (5, 5))
    h, w = g.shape
    hb, wb = h // block, w // block
    if hb < 4 or wb < 4:
        return None, []

    def blocks(a):
        return a[: hb * block, : wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3)).ravel()

    bm, bv = blocks(mean_map), blocks(var_map)
    pts = []
    edges = np.linspace(20, 235, 12)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (bm >= lo) & (bm < hi)
        if sel.sum() < 6:
            continue
        # Low percentile keeps flat blocks and rejects texture/edges.
        v = np.percentile(bv[sel], 20)
        if v > 1e-6:
            pts.append((float(bm[sel].mean()), float(v)))
    if len(pts) < 3:
        return None, pts
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    if np.ptp(x) < 0.7:  # not enough intensity spread to judge dependence
        return None, pts
    b = float(np.polyfit(x, y, 1)[0])
    return b, pts


def detect_noise(img):
    g = _gray(img)
    sigma = global_sigma(img)
    impulse_ratio, extreme_share = _impulse_stats(g)
    slope, pts = _variance_slope(g)

    scores = {"salt_pepper": 0.0, "gaussian": 0.0, "speckle": 0.0, "poisson": 0.0}
    if impulse_ratio > 0.004 and extreme_share > 0.6:
        kind = "salt_pepper"
        conf = min(0.99, 0.6 + extreme_share * 0.4)
        scores["salt_pepper"] = conf
    elif sigma < 2.0:
        kind, conf = "none", 0.9
    elif slope is None:
        # Too little mid-tone area (e.g. a bi-level scanned page) to measure signal
        # dependence; fall back to the ratio of impulsive to moderate residuals.
        res = np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16))
        big = int(((res > 100) & ((g == 0) | (g == 255))).sum())
        mid = int(((res > 12) & (res < 60)).sum())
        if big / max(mid, 1) > 0.25:
            kind, conf = "salt_pepper", 0.6
        else:
            kind, conf = "gaussian", 0.45
        scores[kind] = conf
    else:
        targets = {"gaussian": 0.0, "poisson": 1.0, "speckle": 2.0}
        for k, t in targets.items():
            scores[k] = float(np.exp(-((slope - t) ** 2) / (2 * 0.3 ** 2)))
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}
        kind = max(targets, key=lambda k: scores[k])
        conf = scores[kind]

    rec = RECOMMENDED[kind]
    return {
        "type": kind,
        "confidence": round(float(conf), 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "sigma": round(sigma, 2),
        "impulse_ratio": round(impulse_ratio, 4),
        "slope": None if slope is None else round(slope, 2),
        "variance_curve": [[round(m, 1), round(v, 2)] for m, v in pts],
        "recommendation": rec,
    }


def histogram(img, bins=64):
    g = _gray(img)
    h = cv2.calcHist([g], [0], None, [bins], [0, 256]).ravel()
    return (h / h.sum()).round(5).tolist()

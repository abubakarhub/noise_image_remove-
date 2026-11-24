"""Synthetic noise models (Section 5 of the synopsis).

All functions take and return uint8 images (grayscale or BGR) and accept an
optional seed so that every filter is evaluated against identical noise.
"""
import numpy as np

# Parameter ranges exposed to the UI: (label, min, max, step, default)
NOISE_PARAMS = {
    "salt_pepper": {"label": "Noise density", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.08},
    "gaussian": {"label": "Std. deviation σ", "min": 2, "max": 80, "step": 1, "default": 25},
    "speckle": {"label": "Variance", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.08},
    "poisson": {"label": "Photon peak (lower = noisier)", "min": 2, "max": 255, "step": 1, "default": 20},
}

NOISE_NAMES = {
    "salt_pepper": "Salt & Pepper",
    "gaussian": "Gaussian",
    "speckle": "Speckle",
    "poisson": "Poisson",
}


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def salt_pepper(img, density, rng):
    """Impulse noise: a fraction `density` of pixels forced to 0 or 255."""
    out = img.copy()
    mask = rng.random(img.shape[:2])
    salt = mask < density / 2
    pepper = (mask >= density / 2) & (mask < density)
    out[salt] = 255
    out[pepper] = 0
    return out


def gaussian(img, sigma, rng):
    """Additive zero-mean Gaussian noise: g = f + n, n ~ N(0, sigma^2)."""
    n = rng.normal(0.0, sigma, img.shape)
    return _to_uint8(img.astype(np.float64) + n)


def speckle(img, var, rng):
    """Multiplicative noise: g = f + n*f, n ~ N(0, var)."""
    f = img.astype(np.float64)
    n = rng.normal(0.0, np.sqrt(var), img.shape)
    return _to_uint8(f + n * f)


def poisson(img, peak, rng):
    """Shot noise: photon counts ~ Poisson(f * peak / 255)."""
    f = img.astype(np.float64) / 255.0
    counts = rng.poisson(f * peak)
    return _to_uint8(counts / peak * 255.0)


_FUNCS = {"salt_pepper": salt_pepper, "gaussian": gaussian, "speckle": speckle, "poisson": poisson}


def add_noise(img, kind, level=None, seed=0):
    if kind not in _FUNCS:
        raise ValueError(f"Unknown noise type: {kind}")
    if level is None:
        level = NOISE_PARAMS[kind]["default"]
    rng = np.random.default_rng(seed)
    return _FUNCS[kind](img, float(level), rng)

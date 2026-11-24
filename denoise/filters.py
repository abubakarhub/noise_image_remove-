"""Spatial-domain denoising filters (Section 7 of the synopsis), via OpenCV."""
import cv2

FILTER_NAMES = {
    "mean": "Mean",
    "median": "Median",
    "gaussian": "Gaussian",
    "bilateral": "Bilateral",
}


def _odd(k):
    k = max(1, int(k))
    return k if k % 2 else k + 1


def apply_filter(img, kind, ksize=5, sigma=0.0, sigma_color=75.0, sigma_space=75.0):
    """Apply one filter.

    ksize        neighbourhood size (forced odd); for bilateral it is the diameter d
    sigma        Gaussian std. deviation (0 lets OpenCV derive it from ksize)
    sigma_color  bilateral range (intensity) sigma
    sigma_space  bilateral spatial sigma
    """
    k = _odd(ksize)
    if kind == "mean":
        return cv2.blur(img, (k, k))
    if kind == "median":
        return cv2.medianBlur(img, k)
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (k, k), float(sigma))
    if kind == "bilateral":
        return cv2.bilateralFilter(img, k, float(sigma_color), float(sigma_space))
    raise ValueError(f"Unknown filter: {kind}")

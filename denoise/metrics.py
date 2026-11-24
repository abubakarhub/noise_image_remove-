"""Full-reference quality metrics (Section 10 of the synopsis)."""
import cv2
import numpy as np


def mse(ref, img):
    d = ref.astype(np.float64) - img.astype(np.float64)
    return float(np.mean(d * d))


def psnr(ref, img, max_val=255.0):
    m = mse(ref, img)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / m))


def _ssim_channel(x, y, data_range=255.0, win=7, k1=0.01, k2=0.03):
    """SSIM (Wang et al., 2004) with a 7x7 uniform window and sample covariance.

    Same definition and defaults as skimage.metrics.structural_similarity, written with
    OpenCV so the app does not need SciPy/scikit-image (keeps the serverless bundle small).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n = win * win
    cov_norm = n / (n - 1)
    box = lambda a: cv2.blur(a, (win, win), borderType=cv2.BORDER_REFLECT)
    ux, uy = box(x), box(y)
    vx = cov_norm * (box(x * x) - ux * ux)
    vy = cov_norm * (box(y * y) - uy * uy)
    vxy = cov_norm * (box(x * y) - ux * uy)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    p = (win - 1) // 2
    return float(s[p:-p, p:-p].mean())


def ssim(ref, img):
    if ref.ndim == 3:
        return float(np.mean([_ssim_channel(ref[..., c], img[..., c]) for c in range(ref.shape[2])]))
    return _ssim_channel(ref, img)


def evaluate(ref, img):
    p = psnr(ref, img)
    return {"mse": mse(ref, img), "psnr": p if np.isfinite(p) else 99.0, "ssim": ssim(ref, img)}

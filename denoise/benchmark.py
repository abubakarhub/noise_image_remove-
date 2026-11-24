"""Systematic comparison: every filter x kernel size on identical noisy input."""
import time

from .analysis import detect_noise
from .filters import apply_filter, FILTER_NAMES
from .metrics import evaluate
from .noise import add_noise, NOISE_PARAMS

KERNELS = (3, 5, 7, 9)


def compare_filters(clean, noisy, kernels=KERNELS, filter_kwargs=None):
    """Return one row per (filter, kernel) with MSE/PSNR/SSIM and run time."""
    filter_kwargs = filter_kwargs or {}
    rows = []
    for f in FILTER_NAMES:
        for k in kernels:
            t0 = time.perf_counter()
            out = apply_filter(noisy, f, k, **filter_kwargs.get(f, {}))
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"filter": f, "ksize": k, "ms": round(ms, 2), **evaluate(clean, out), "_img": out})
    return rows


def best_row(rows, metric="ssim"):
    key = (lambda r: -r["mse"]) if metric == "mse" else (lambda r: r[metric])
    return max(rows, key=key)


def noise_filter_matrix(clean, kernels=(3, 5, 7), seed=0):
    """Best SSIM/PSNR of each filter for each noise type at its default level."""
    out = {}
    for n, p in NOISE_PARAMS.items():
        noisy = add_noise(clean, n, p["default"], seed)
        base = evaluate(clean, noisy)
        rows = compare_filters(clean, noisy, kernels)
        cells = {}
        for f in FILTER_NAMES:
            b = best_row([r for r in rows if r["filter"] == f])
            cells[f] = {k: b[k] for k in ("ksize", "mse", "psnr", "ssim")}
        winner = max(cells, key=lambda f: cells[f]["ssim"])
        from .enhanced import apply_enhanced  # local import avoids a cycle at package load
        info = detect_noise(noisy)
        auto_img, _ = apply_enhanced(noisy, "auto", info)
        auto = dict(evaluate(clean, auto_img), method=info["enhanced"]["filter"], detected=info["type"])
        out[n] = {"noisy": base, "filters": cells, "best": winner, "level": p["default"], "auto": auto}
    return out

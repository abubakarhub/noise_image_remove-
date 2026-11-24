"""FastAPI backend for the Image Noise Removal lab.

Run:  python run.py   (then open http://127.0.0.1:8000)
"""
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from denoise import (FILTER_NAMES, NOISE_NAMES, NOISE_PARAMS, RECOMMENDED, add_noise,
                     apply_filter, detect_noise, evaluate, global_sigma, histogram)
from denoise.benchmark import KERNELS, best_row, compare_filters, noise_filter_matrix
from denoise.enhanced import ENHANCED_NAMES, apply_enhanced

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
WEB = ROOT / "web"
MAX_SIDE = 720       # interactive processing size
MATRIX_SIDE = 384    # the 48-run matrix uses a smaller copy

SAMPLE_INFO = {
    "cameraman": "Cameraman - classic grayscale test image",
    "astronaut": "Astronaut - colour portrait",
    "coffee": "Coffee - colour, fine texture",
    "coins": "Coins - grayscale, sharp edges",
    "moon": "Moon - smooth surface, low contrast",
    "document": "Scanned notice - document processing",
}

app = FastAPI(title="Image Noise Removal Lab")


# ---------------------------------------------------------------- helpers
def _fit(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _normalise(img):
    """uint8, grayscale or 3-channel BGR (alpha dropped, grey-in-colour collapsed)."""
    if img is None:
        raise HTTPException(400, "Could not decode the image.")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        b, g, r = cv2.split(img)
        if np.array_equal(b, g) and np.array_equal(g, r):
            img = b
    return img


async def _load(file: UploadFile | None, sample: str | None, side=MAX_SIDE):
    if file is not None and file.filename:
        data = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    else:
        name = sample or "cameraman"
        if name not in SAMPLE_INFO:
            raise HTTPException(404, f"Unknown sample {name}")
        img = cv2.imread(str(SAMPLES / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    return _fit(_normalise(img), side)


def _b64(img, fmt=".png"):
    ok, buf = cv2.imencode(fmt, img)
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf).decode()


def _diff_map(a, b):
    """|a-b| amplified and colour-mapped, for the 'residual' view."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    d = cv2.absdiff(ga, gb).astype(np.float32)
    d = np.clip(d * 4, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def _filter_kwargs(sigma, sigma_color, sigma_space):
    return {"gaussian": {"sigma": sigma},
            "bilateral": {"sigma_color": sigma_color, "sigma_space": sigma_space}}


def _method_info(key, analysis):
    """Which enhanced method actually ran (Auto resolves to one) and with what strength."""
    applied = analysis["enhanced"]["filter"] if key == "auto" else key
    if applied == "none":
        label = "Auto → no filtering (image looks clean)"
    else:
        label = f"Auto → {ENHANCED_NAMES[applied]}" if key == "auto" else ENHANCED_NAMES[applied]
    p = analysis.get("params") or {}
    strength = ", ".join(f"{k} {v}" for k, v in p.items())
    return {"key": key, "applied": applied, "label": label, "enhanced": True,
            "strength": strength, "why": analysis["enhanced"]["why"]}


# ---------------------------------------------------------------- API
@app.get("/api/config")
def config():
    return {
        "noises": {k: {"name": NOISE_NAMES[k], **v} for k, v in NOISE_PARAMS.items()},
        "filters": FILTER_NAMES,
        "enhanced": ENHANCED_NAMES,
        "kernels": list(KERNELS),
        "samples": SAMPLE_INFO,
        "recommended": RECOMMENDED,
    }


@app.get("/api/sample/{name}")
def sample(name: str):
    if name not in SAMPLE_INFO:
        raise HTTPException(404)
    return FileResponse(SAMPLES / f"{name}.png")


@app.post("/api/process")
async def process(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    noise: str = Form("gaussian"), level: float = Form(25), seed: int = Form(0),
    filter: str = Form("median"), ksize: int = Form(5), sigma: float = Form(0),
    sigma_color: float = Form(75), sigma_space: float = Form(75),
):
    """Reference experiment: clean -> add noise -> analyse -> filter -> evaluate."""
    clean = await _load(file, sample)
    if noise not in NOISE_PARAMS or (filter not in FILTER_NAMES and filter not in ENHANCED_NAMES):
        raise HTTPException(400, "Unknown noise or filter")
    noisy = add_noise(clean, noise, level, seed)
    analysis = detect_noise(noisy)
    if filter in ENHANCED_NAMES:
        out, _ = apply_enhanced(noisy, filter, analysis)
        method = _method_info(filter, analysis)
    else:
        out = apply_filter(noisy, filter, ksize, **_filter_kwargs(sigma, sigma_color, sigma_space).get(filter, {}))
        method = {"key": filter, "label": f"{FILTER_NAMES[filter]} {ksize}×{ksize}", "enhanced": False}
    return {
        "method": method,
        "size": [clean.shape[1], clean.shape[0]],
        "color": clean.ndim == 3,
        "images": {"original": _b64(clean), "noisy": _b64(noisy), "denoised": _b64(out),
                   "residual": _b64(_diff_map(clean, out), ".jpg")},
        "metrics": {"noisy": evaluate(clean, noisy), "denoised": evaluate(clean, out)},
        "histograms": {"original": histogram(clean), "noisy": histogram(noisy), "denoised": histogram(out)},
        "analysis": analysis,
    }


@app.post("/api/detect")
async def detect(file: UploadFile | None = File(None), sample: str | None = Form(None)):
    """No-reference mode: user's own noisy photo -> detect type -> apply recommended filter."""
    noisy = await _load(file, sample)
    info = detect_noise(noisy)
    out, _ = apply_enhanced(noisy, "auto", info)
    return {
        "analysis": info,
        "method": _method_info("auto", info),
        "images": {"noisy": _b64(noisy), "denoised": _b64(out), "residual": _b64(_diff_map(noisy, out), ".jpg")},
        "sigma_after": round(global_sigma(out), 2),
        "histograms": {"noisy": histogram(noisy), "denoised": histogram(out)},
    }


@app.post("/api/compare")
async def compare(
    file: UploadFile | None = File(None), sample: str | None = Form(None),
    noise: str = Form("gaussian"), level: float = Form(25), seed: int = Form(0),
    sigma: float = Form(0), sigma_color: float = Form(75), sigma_space: float = Form(75),
):
    """Phase 6: every filter x kernel size on the same noisy image."""
    clean = _fit(await _load(file, sample), 512)
    noisy = add_noise(clean, noise, level, seed)
    rows = compare_filters(clean, noisy, KERNELS, _filter_kwargs(sigma, sigma_color, sigma_space))
    best = {f: best_row([r for r in rows if r["filter"] == f]) for f in FILTER_NAMES}
    overall = best_row(rows)
    info = detect_noise(noisy)
    auto_img, _ = apply_enhanced(noisy, "auto", info)
    auto = {**evaluate(clean, auto_img), "image": _b64(auto_img), "method": _method_info("auto", info)}
    return {
        "auto": auto,
        "noisy": evaluate(clean, noisy),
        "noisy_image": _b64(noisy),
        "rows": [{k: v for k, v in r.items() if k != "_img"} for r in rows],
        "best": {f: {"ksize": r["ksize"], "ssim": r["ssim"], "psnr": r["psnr"], "mse": r["mse"],
                     "image": _b64(r["_img"])} for f, r in best.items()},
        "overall": {"filter": overall["filter"], "ksize": overall["ksize"], "ssim": overall["ssim"],
                    "psnr": overall["psnr"]},
    }


@app.post("/api/matrix")
async def matrix(file: UploadFile | None = File(None), sample: str | None = Form(None), seed: int = Form(0)):
    """Experimental version of Table 2: best filter per noise type."""
    clean = await _load(file, sample, MATRIX_SIDE)
    return noise_filter_matrix(clean, seed=seed)


# ---------------------------------------------------------------- static site
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

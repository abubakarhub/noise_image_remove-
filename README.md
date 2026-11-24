# Image Noise Removal using Digital Image Processing Filters

Minor Project (ARP-455) · B.Tech IIoT · USAR, GGSIPU East Delhi Campus
Abubakar Siddiqui (08019051723) · Supervisor: Ms. Anuj Kumar

Four noise models (salt-and-pepper, Gaussian, speckle, Poisson) are added to clean
images, removed with four spatial filters (mean, median, Gaussian, bilateral), and
evaluated with MSE, PSNR and SSIM. A web front end ("Denoise Lab") runs the real
Python/OpenCV pipeline interactively.

## Run

```
pip install -r requirements.txt   # add requirements-dev.txt for plots
python run.py                # or double-click start.bat → opens http://127.0.0.1:8000
python experiments/run_experiments.py          # full batch → results/
python experiments/run_experiments.py --quick  # 2 images, default levels
```

## Structure

| Path | Purpose |
|---|---|
| `denoise/noise.py` | Noise models (NumPy): salt & pepper, Gaussian `g=f+n`, speckle `g=f+n·f`, Poisson |
| `denoise/filters.py` | `cv2.blur`, `medianBlur`, `GaussianBlur`, `bilateralFilter` |
| `denoise/metrics.py` | MSE, PSNR, SSIM (OpenCV implementation, identical to scikit-image) |
| `denoise/analysis.py` | No-reference noise detection + recommended filter |
| `denoise/benchmark.py` | Filter × kernel comparison and noise × filter matrix |
| `server.py` | FastAPI: `/api/process`, `/api/detect`, `/api/compare`, `/api/matrix` |
| `web/` | Front end (HTML/CSS/JS, no build step) |
| `experiments/run_experiments.py` | Batch run → `results/results.csv`, `summary.md`, plots |
| `samples/` | Test images (incl. a scanned notice for the document-processing case) |

## Pipeline

Input image → Noise addition → Noise analysis → Filtering → Denoised image → Quality evaluation → Comparison

## Noise detection (objective 1)

Without a clean reference the system classifies the noise:

1. **Impulse test**: isolated 0/255 pixels that disagree with their 3×3 median mean salt-and-pepper.
2. **Signal-dependence test**: local variance (Immerkær operator) is regressed against local mean
   on a log–log scale. Slope ≈ 0 means Gaussian, ≈ 1 means Poisson, ≈ 2 means speckle.

On the natural sample images it identifies the added noise correctly in all tested cases. On bi-level
documents (almost no mid-tones) it falls back to a weaker heuristic and reports lower confidence.

## Results so far (`results/summary.md`)

Mean SSIM / PSNR at each filter's best kernel, over 6 images × 3 noise levels:

| Noise | Mean | Median | Gaussian | Bilateral | Best | Synopsis expected |
|---|---|---|---|---|---|---|
| Salt & Pepper | 0.588 / 22.7 | **0.887 / 28.8** | 0.568 / 23.1 | 0.305 / 17.3 | Median | Median ✓ |
| Gaussian | 0.701 / 25.4 | 0.696 / 25.1 | **0.719** / 26.0 | 0.707 / **27.3** | Gaussian (SSIM) / Bilateral (PSNR) | Gaussian / Bilateral ✓ |
| Speckle | 0.677 / 24.0 | 0.653 / 23.9 | **0.679 / 24.4** | 0.584 / 23.5 | Gaussian | Median / Bilateral ✗ |
| Poisson | 0.671 / 24.0 | 0.573 / 23.2 | **0.672 / 24.3** | 0.569 / 23.6 | Gaussian | Gaussian / Bilateral ✓ |

Three of the four predictions in Table 2 hold. For speckle, Gaussian smoothing beat median and bilateral.
Bilateral at default σ (75) under-smooths strong multiplicative noise, and median's advantage is
limited to impulse outliers.

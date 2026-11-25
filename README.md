# Image Noise Removal using Digital Image Processing Filters

Minor Project (ARP-455) · B.Tech IIoT · USAR, GGSIPU East Delhi Campus
Abubakar Siddiqui (08019051723) · Supervisor: Ms. Anuj Kumar

Four noise models (salt-and-pepper, Gaussian, speckle, Poisson) are added to clean
images, removed with four spatial filters (mean, median, Gaussian, bilateral), and
evaluated with MSE, PSNR and SSIM. A web front end ("Denoise Lab") runs the real
Python/OpenCV pipeline interactively.

On top of the four classic filters, a noise-aware **Auto** mode detects the noise type
without a clean reference, measures its strength, and applies a matching self-tuning
method. It beats the best classic filter (chosen with the clean image) by +0.057 SSIM and
+3.4 dB PSNR on average. Noise detection is 95.9% accurate on the samples. Full numbers are in
[`results/accuracy.md`](results/accuracy.md).

Live: https://noise-image-remove-hpqk.vercel.app/

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
| `denoise/analysis.py` | No-reference noise detection, noise-strength estimation, recommended filter |
| `denoise/enhanced.py` | Noise-aware methods: adaptive median, non-local means, VST + NLM, Auto |
| `denoise/benchmark.py` | Filter × kernel comparison and noise × filter matrix |
| `server.py` | FastAPI: `/api/process`, `/api/detect`, `/api/compare`, `/api/matrix` |
| `web/` | Front end (HTML/CSS/JS, no build step) |
| `experiments/run_experiments.py` | Batch run → `results/results.csv`, `summary.md`, plots |
| `experiments/eval_detection.py` | Detection accuracy + confusion matrix |
| `experiments/eval_denoise.py` | Auto vs. best classic filter |
| `samples/` | Test images (incl. a scanned notice for the document-processing case) |

## Pipeline

Input image → Noise addition → Noise analysis → Filtering → Denoised image → Quality evaluation → Comparison

## Noise detection (objective 1)

Without a clean reference the system classifies the noise and estimates its strength:

1. **Impulse test**: salt-and-pepper pixels are isolated spikes (differ from all 8 neighbours)
   whose value is exactly 0 or 255. Other noise types produce no such spikes, so the ratio of
   0/255 spikes to other spikes separates them. 100% on all test images.
2. **Noise-level function**: the local variance (Immerkær operator, 8×8 blocks, 20th percentile
   per intensity bin to skip edges and texture) is measured against intensity, per colour channel.
3. **Forward models with clipping**: for Gaussian (var = σ²), Poisson (var = 255·f/peak) and
   speckle (var = c·f²) the exact mean and variance *after clipping to 0–255* are computed
   (censored normal / exact Poisson). The model and strength that best fit the measured curve
   win. Clipped dark/bright regions become evidence instead of a source of error. Saturated
   white areas give a separate one-sided estimate.
4. **Clean test**: if the fitted noise at mid-grey is below 3 grey levels, the image is reported
   as clean and Auto leaves it untouched.

| | Old detector | Now |
|---|---|---|
| Project samples (246 cases) | 87.8% | **95.9%** |
| 15 validation images (615 cases) | 74.0% | **94.8%** |

The remaining errors are mostly on bi-level images (scanned page, silhouettes). With only black
and white present, speckle and Poisson predict the same curve.

## Noise-aware denoising (Auto)

| Detected | Method | Strength from |
|---|---|---|
| Salt & pepper | Adaptive median: only impulse pixels are replaced, by the median of their clean neighbours (window grows 3→15) | impulse mask |
| Gaussian | Non-local means (Buades et al., 2005) with a clipping-aware inverse | estimated σ |
| Poisson / speckle | Variance-stabilising transform built from the fitted noise curve (Anscombe / log), NLM, exact unbiased inverse (Makitalo & Foi, 2011) including clipping | estimated peak / variance |
| Clean | none | — |

The NLM strength factor depends on the noise severity, from tables tuned on the validation
images. Colour images are processed per channel.

| Noise | Best classic (oracle) SSIM / PSNR | Auto SSIM / PSNR | Auto wins |
|---|---|---|---|
| Salt & Pepper | 0.887 / 28.83 dB | **0.986 / 37.89 dB** | 18/18 |
| Gaussian | 0.765 / 27.53 dB | **0.817 / 29.48 dB** | 15/18 |
| Speckle | 0.703 / 24.81 dB | **0.726 / 25.75 dB** | 13/18 |
| Poisson | 0.708 / 25.28 dB | **0.763 / 26.99 dB** | 17/18 |

## Classic filter results (`results/summary.md`)

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

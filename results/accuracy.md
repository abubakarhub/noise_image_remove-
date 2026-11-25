# Accuracy of the improved pipeline

Reproduce with `python experiments/eval_detection.py` and `python experiments/eval_denoise.py`.

## 1. Noise detection (no clean reference)

6 sample images x 4 noise types x 5 levels x 2 seeds, plus each clean image (expected: "none").
Levels: salt & pepper 0.01-0.4, Gaussian sigma 5-75, speckle 0.01-0.4, Poisson peak 200-3.

**Project samples** (previous detector: 87.8%)

```
overall accuracy: 236/246 = 95.9%
truth / predicted       none  salt_pep  gaussian   speckle   poisson   acc
none                       6         0         0         0         0  100%
salt_pepper                0        60         0         0         0  100%
gaussian                   0         0        58         0         2  97%
speckle                    0         0         2        56         2  93%
poisson                    0         0         4         0        56  93%
```

**15 extra validation images** (`python experiments/make_validation_set.py` → `validation/`)
(scikit-image data set: chelsea, rocket, page, text, brick, grass, gravel, clock, hubble,
immunohistochemistry, cell, horse, logo, colorwheel, retina; previous detector: 74.0%)

```
overall accuracy: 583/615 = 94.8%
truth / predicted       none  salt_pep  gaussian   speckle   poisson   acc
none                      12         0         3         0         0  80%
salt_pepper                0       150         0         0         0  100%
gaussian                   0         0       150         0         0  100%
speckle                    0         0         0       124        26  83%
poisson                    0         0         3         0       147  98%
```

Remaining errors are mostly bi-level images (scanned page, horse/logo silhouettes): with only
black and white present, speckle and Poisson predict the same curve and cannot be separated.

## 2. Denoising: Auto (noise-aware) vs. the best classic filter

Same images and levels as `run_experiments.py`. The classic column is the best of
mean/median/Gaussian/bilateral x kernel 3/5/7/9 chosen *with the clean image* (an oracle
upper bound); Auto only sees the noisy image.

| Noise | Best classic filter (oracle) SSIM / PSNR | Auto (noise-aware) SSIM / PSNR | SSIM gain | PSNR gain | Auto wins |
|---|---|---|---|---|---|
| Salt & Pepper | 0.887 / 28.83 dB | **0.986 / 37.89 dB** | +0.100 | +9.06 dB | 18/18 |
| Gaussian | 0.765 / 27.53 dB | **0.817 / 29.48 dB** | +0.052 | +1.94 dB | 15/18 |
| Speckle | 0.703 / 24.81 dB | **0.726 / 25.75 dB** | +0.022 | +0.94 dB | 13/18 |
| Poisson | 0.708 / 25.28 dB | **0.763 / 26.99 dB** | +0.055 | +1.71 dB | 17/18 |
| **All** | 0.766 / 26.61 dB | **0.823 / 30.03 dB** | +0.057 | +3.41 dB | |

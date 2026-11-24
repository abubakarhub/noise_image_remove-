# Experimental results

Mean SSIM / PSNR of each filter (best kernel per run), averaged over all images and noise levels.

| Noise | Mean | Median | Gaussian | Bilateral | Best | Synopsis expected | Match |
|---|---|---|---|---|---|---|---|
| Salt & Pepper | 0.588 / 22.7 dB | 0.887 / 28.8 dB | 0.568 / 23.1 dB | 0.305 / 17.3 dB | **Median** | Median | yes |
| Gaussian | 0.701 / 25.4 dB | 0.696 / 25.1 dB | 0.719 / 26.0 dB | 0.707 / 27.3 dB | **Gaussian** | Gaussian / Bilateral | yes |
| Speckle | 0.677 / 24.0 dB | 0.653 / 23.9 dB | 0.679 / 24.4 dB | 0.584 / 23.5 dB | **Gaussian** | Median / Bilateral | no |
| Poisson | 0.671 / 24.0 dB | 0.573 / 23.2 dB | 0.672 / 24.3 dB | 0.569 / 23.6 dB | **Gaussian** | Gaussian / Bilateral | yes |

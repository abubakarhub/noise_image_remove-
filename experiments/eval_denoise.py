"""Enhanced (noise-aware) denoising vs. the best classic filter.

For every image x noise type x level (the same levels as run_experiments.py):
  classic  = best of mean/median/Gaussian/bilateral x kernels 3,5,7,9, picked with the
             clean image in hand (an oracle choice - an upper bound for the classic filters)
  auto     = detect noise on the noisy image only -> enhanced method with estimated strength

    python experiments/eval_denoise.py [--dir FOLDER] [--out results/enhanced_summary.md]
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from denoise import add_noise, evaluate  # noqa: E402
from denoise.benchmark import KERNELS, compare_filters, best_row  # noqa: E402
from denoise.enhanced import apply_enhanced  # noqa: E402
from eval_detection import load  # noqa: E402
from run_experiments import LEVELS  # noqa: E402

NAMES = {"salt_pepper": "Salt & Pepper", "gaussian": "Gaussian", "speckle": "Speckle", "poisson": "Poisson"}


def run(folder=ROOT / "samples", seed=0, quiet=False):
    res = defaultdict(list)  # noise -> [(classic ssim, classic psnr, auto ssim, auto psnr, ms)]
    for path in sorted(Path(folder).glob("*.png")):
        clean = load(path)
        for noise, levels in LEVELS.items():
            for lv in levels:
                noisy = add_noise(clean, noise, lv, seed)
                rows = compare_filters(clean, noisy, KERNELS)
                c = best_row(rows, "ssim")
                t0 = time.perf_counter()
                out, info = apply_enhanced(noisy, "auto")
                ms = (time.perf_counter() - t0) * 1000
                a = evaluate(clean, out)
                res[noise].append((c["ssim"], c["psnr"], a["ssim"], a["psnr"], ms))
                if not quiet:
                    print(f"{path.stem:12s} {noise:11s} {lv:<6} classic {c['filter']:9s}{c['ksize']} "
                          f"SSIM {c['ssim']:.3f} PSNR {c['psnr']:5.2f} | auto[{info['type'][:6]}] "
                          f"SSIM {a['ssim']:.3f} PSNR {a['psnr']:5.2f}  {ms:5.0f} ms", flush=True)
    return res


def table(res):
    lines = ["| Noise | Best classic filter (oracle) SSIM / PSNR | Auto (noise-aware) SSIM / PSNR | SSIM gain | PSNR gain | Auto wins |",
             "|---|---|---|---|---|---|"]
    allc, alla = [], []
    for noise, r in res.items():
        a = np.array(r)
        cs, cp, as_, ap = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()
        wins = int((a[:, 2] > a[:, 0]).sum())
        lines.append(f"| {NAMES[noise]} | {cs:.3f} / {cp:.2f} dB | **{as_:.3f} / {ap:.2f} dB** | "
                     f"{as_ - cs:+.3f} | {ap - cp:+.2f} dB | {wins}/{len(a)} |")
        allc.append(a[:, :2]); alla.append(a[:, 2:4])
    c, a = np.vstack(allc).mean(0), np.vstack(alla).mean(0)
    lines.append(f"| **All** | {c[0]:.3f} / {c[1]:.2f} dB | **{a[0]:.3f} / {a[1]:.2f} dB** | {a[0] - c[0]:+.3f} | {a[1] - c[1]:+.2f} dB | |")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    folder = args[args.index("--dir") + 1] if "--dir" in args else ROOT / "samples"
    res = run(folder, quiet="-q" in args)
    t = table(res)
    print(t)
    if "--out" in args:
        Path(args[args.index("--out") + 1]).write_text(t + "\n", encoding="utf-8")

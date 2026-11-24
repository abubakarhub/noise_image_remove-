"""Batch experiment for the report (Phase 6: performance evaluation & comparison).

Runs every sample image x noise type x 3 noise levels x 4 filters x 4 kernel sizes,
and writes:
    results/results.csv          every run (MSE, PSNR, SSIM, time)
    results/summary.md           best filter per noise type + Table 2 check
    results/plots/*.png          metric-vs-kernel plots and image grids (needs matplotlib)

Usage:  python experiments/run_experiments.py [--quick]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoise import FILTER_NAMES, NOISE_NAMES, NOISE_PARAMS, add_noise, apply_filter, evaluate  # noqa: E402
from denoise.benchmark import KERNELS, compare_filters  # noqa: E402

LEVELS = {
    "salt_pepper": [0.05, 0.10, 0.20],
    "gaussian": [15, 25, 40],
    "speckle": [0.04, 0.08, 0.16],
    "poisson": [60, 20, 8],
}
EXPECTED = {  # Table 2 of the synopsis
    "salt_pepper": ["median"], "gaussian": ["gaussian", "bilateral"],
    "speckle": ["median", "bilateral"], "poisson": ["gaussian", "bilateral"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two images, default level only")
    args = ap.parse_args()

    out_dir = ROOT / "results"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    images = sorted((ROOT / "samples").glob("*.png"))
    if args.quick:
        images = images[:2]

    rows = []
    for path in images:
        clean = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        print(f"{path.stem}: {clean.shape}")
        for noise, levels in LEVELS.items():
            for level in ([NOISE_PARAMS[noise]["default"]] if args.quick else levels):
                noisy = add_noise(clean, noise, level, seed=0)
                base = evaluate(clean, noisy)
                rows.append({"image": path.stem, "noise": noise, "level": level, "filter": "none", "ksize": 0,
                             "ms": 0, **base})
                for r in compare_filters(clean, noisy, KERNELS):
                    r.pop("_img")
                    rows.append({"image": path.stem, "noise": noise, "level": level, **r})

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "noise", "level", "filter", "ksize", "mse", "psnr", "ssim", "ms"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {csv_path} ({len(rows)} rows)")

    write_summary(rows, out_dir / "summary.md")
    try:
        make_plots(rows, out_dir / "plots", images)
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install matplotlib)")


def write_summary(rows, path):
    # average SSIM/PSNR of each filter's best kernel, per noise type, over images and levels
    best = defaultdict(lambda: defaultdict(list))  # noise -> filter -> [(ssim, psnr, k)]
    groups = defaultdict(list)
    for r in rows:
        if r["filter"] != "none":
            groups[(r["image"], r["noise"], r["level"], r["filter"])].append(r)
    for (img, noise, level, f), rs in groups.items():
        b = max(rs, key=lambda r: r["ssim"])
        best[noise][f].append((b["ssim"], b["psnr"], b["ksize"]))

    lines = ["# Experimental results", "",
             "Mean SSIM / PSNR of each filter (best kernel per run), averaged over all images and noise levels.", "",
             "| Noise | " + " | ".join(FILTER_NAMES.values()) + " | Best | Synopsis expected | Match |",
             "|---" * (len(FILTER_NAMES) + 4) + "|"]
    for noise in LEVELS:
        means = {f: (sum(s for s, _, _ in v) / len(v), sum(p for _, p, _ in v) / len(v)) for f, v in best[noise].items()}
        win = max(means, key=lambda f: means[f][0])
        cells = [f"{means[f][0]:.3f} / {means[f][1]:.1f} dB" for f in FILTER_NAMES]
        exp = " / ".join(FILTER_NAMES[f] for f in EXPECTED[noise])
        ok = "yes" if win in EXPECTED[noise] else "no"
        lines.append(f"| {NOISE_NAMES[noise]} | " + " | ".join(cells) + f" | **{FILTER_NAMES[win]}** | {exp} | {ok} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("\n".join(lines))


def make_plots(rows, plot_dir, images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # metric vs kernel size, per noise type (default level, averaged over images)
    for metric in ("ssim", "psnr"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, noise in zip(axes, LEVELS):
            lvl = NOISE_PARAMS[noise]["default"]
            sel = [r for r in rows if r["noise"] == noise and (r["level"] == lvl or len(LEVELS[noise]) == 1)]
            if not sel:
                sel = [r for r in rows if r["noise"] == noise]
            for f in FILTER_NAMES:
                ys = [sum(r[metric] for r in sel if r["filter"] == f and r["ksize"] == k) /
                      max(1, sum(1 for r in sel if r["filter"] == f and r["ksize"] == k)) for k in KERNELS]
                ax.plot(KERNELS, ys, marker="o", label=FILTER_NAMES[f])
            ax.set_title(NOISE_NAMES[noise])
            ax.set_xlabel("kernel size")
            ax.set_xticks(KERNELS)
            ax.grid(alpha=.3)
        axes[0].set_ylabel(metric.upper())
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_vs_kernel.png", dpi=150)
        plt.close(fig)

    # visual grid for the first image: noisy + each filter at k=5
    clean = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
    cmap = "gray" if clean.ndim == 2 else None
    show = (lambda im: im) if clean.ndim == 2 else (lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for row, noise in zip(axes, LEVELS):
        noisy = add_noise(clean, noise, None, 0)
        panels = [("Original", clean), (NOISE_NAMES[noise], noisy)] + \
                 [(FILTER_NAMES[f], apply_filter(noisy, f, 5)) for f in FILTER_NAMES]
        for ax, (title, im) in zip(row, panels):
            ax.imshow(show(im), cmap=cmap, vmin=0, vmax=255)
            m = evaluate(clean, im)
            ax.set_title(f"{title}\nPSNR {m['psnr']:.1f}  SSIM {m['ssim']:.3f}" if title != "Original" else title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / f"grid_{images[0].stem}.png", dpi=120)
    plt.close(fig)
    print(f"wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()

r"""Accuracy of the no-reference noise detector.

Every sample image x noise type x 5 levels x 2 seeds, plus each clean image
(expected answer "none"). Prints accuracy per noise type and a confusion matrix.

    python experiments/eval_detection.py [-v] [--dir FOLDER]
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from denoise import add_noise, detect_noise  # noqa: E402

LEVELS = {
    "salt_pepper": [0.01, 0.03, 0.08, 0.2, 0.4],
    "gaussian": [5, 10, 25, 50, 75],
    "speckle": [0.01, 0.03, 0.08, 0.2, 0.4],
    "poisson": [200, 60, 20, 8, 3],
}
SEEDS = (0, 1)
CLASSES = ["none", "salt_pepper", "gaussian", "speckle", "poisson"]


def load(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3:
        b, g, r = cv2.split(img)
        if (b == g).all() and (g == r).all():
            img = b
    return img


def run(folder=ROOT / "samples", verbose=False):
    images = sorted(Path(folder).glob("*.png"))
    conf = defaultdict(Counter)
    misses = []
    for path in images:
        name, clean = path.stem, load(path)
        cases = [("none", None, 0, clean)]
        for kind, levels in LEVELS.items():
            for lv in levels:
                for s in SEEDS:
                    cases.append((kind, lv, s, add_noise(clean, kind, lv, s)))
        for truth, lv, s, img in cases:
            got = detect_noise(img)["type"]
            conf[truth][got] += 1
            if got != truth:
                misses.append((name, truth, lv, s, got))
    total = sum(sum(c.values()) for c in conf.values())
    right = sum(conf[k][k] for k in conf)
    print(f"overall accuracy: {right}/{total} = {right / total:.1%}")
    print("truth / predicted".ljust(18) + "".join(c[:8].rjust(10) for c in CLASSES) + "   acc")
    for t in CLASSES:
        n = sum(conf[t].values())
        print(t.ljust(18) + "".join(str(conf[t][c]).rjust(10) for c in CLASSES) + f"  {conf[t][t] / n:.0%}")
    if verbose:
        for m in misses:
            print("  miss:", *m)
    return right / total, misses


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[a.index("--dir") + 1] if "--dir" in a else ROOT / "samples", verbose="-v" in a)

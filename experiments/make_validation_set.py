"""Write the 15 validation images (scikit-image data set) to validation/ at <= 512 px.

These images are not part of the app; they are used only to tune/check the detector and the
enhanced filters on pictures other than the project samples:

    pip install scikit-image
    python experiments/make_validation_set.py
    python experiments/eval_detection.py --dir validation
"""
from pathlib import Path

import cv2
import numpy as np
import skimage.data as data

NAMES = ["chelsea", "rocket", "page", "text", "brick", "grass", "gravel", "clock", "hubble_deep_field",
         "immunohistochemistry", "cell", "horse", "logo", "colorwheel", "retina"]
OUT = Path(__file__).resolve().parents[1] / "validation"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for n in NAMES:
        im = np.asarray(getattr(data, n)())
        if im.dtype == bool:
            im = im.astype(np.uint8) * 255
        if im.dtype != np.uint8:
            im = cv2.normalize(im.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[..., :3]
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        h, w = im.shape[:2]
        s = 512 / max(h, w)
        if s < 1:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{n}.png"), im)
        print("wrote", n, im.shape)

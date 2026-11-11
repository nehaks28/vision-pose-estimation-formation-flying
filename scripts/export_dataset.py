"""Export the synthetic dataset to CSV so it can be opened in Excel / pandas.

    python scripts/export_dataset.py              # full dataset -> ../pose-dataset/{train,val,test}.csv
    python scripts/export_dataset.py --n 5000     # small sample

Same generator and seeds as train_ann.py, so train/val/test match what the ANN saw.
Columns: true pose (x, y, z, roll), class label (0..4799), then for each of the 14
features: u (px offset from image centre), v, occluded (1 = not detected).
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from poseest import dataset, aircraft as A

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="rows per split; 0 = full sizes used for training (300k/30k/30k)")
a = ap.parse_args()
OUT = os.path.abspath(os.path.join(ROOT, "..", "pose-dataset"))   # separate folder next to the project
ap2 = None
os.makedirs(OUT, exist_ok=True)

header = ["x_m", "y_m", "z_m", "roll_deg", "label"]
for name in A.FEATURE_NAMES:
    header += [f"{name}_u", f"{name}_v", f"{name}_occluded"]

FULL = {"train": 300_000, "val": 30_000, "test": 30_000}
for split, seed in [("train", 11), ("val", 22), ("test", 33)]:
    n = a.n or FULL[split]
    X, y, pose = dataset.make_split(n, seed)
    X = X.reshape(len(X), A.N_FEAT, 3).copy()
    X[..., 0] *= 640          # undo normalisation -> pixels
    X[..., 1] *= 480
    rows = np.hstack([pose, y[:, None], X.reshape(len(X), -1)])
    path = f"{OUT}/{split}.csv"
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="",
               fmt=["%.2f"] * 4 + ["%d"] + ["%.2f", "%.2f", "%d"] * A.N_FEAT)
    print(f"wrote {path}  ({len(rows)} rows, {len(header)} columns)")

"""Load the exported dataset CSVs (see the separate `pose-dataset/` folder) into the
arrays the ANN expects."""
import os
import numpy as np
from . import config as C, aircraft as A


def load_csv(path):
    """-> X (N,42) float32 network features, y (N,) int labels, pose (N,4) true poses."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Pass --data-dir pointing at the pose-dataset folder.")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    pose = raw[:, :4].astype(np.float64)
    y = raw[:, 4].astype(np.int64)
    f = raw[:, 5:].reshape(len(raw), A.N_FEAT, 3).copy()
    f[..., 0] /= C.IMG_W / 2          # same normalisation as aircraft.to_features
    f[..., 1] /= C.IMG_H / 2
    return f.reshape(len(raw), -1).astype(np.float32), y, pose


def load_split(data_dir, split):
    return load_csv(os.path.join(data_dir, f"{split}.csv"))

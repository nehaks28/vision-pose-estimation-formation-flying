"""Synthetic dataset generation: random relative poses -> noisy detected features -> label."""
import numpy as np
from . import config as C, aircraft as A, discretize as D


def sample_poses(n, rng):
    """Uniform in the discretised pose box, rejecting poses closer than MIN_RANGE."""
    out = np.empty((0, 4))
    while len(out) < n:
        p = rng.uniform(C.POSE_MIN, C.POSE_MAX, size=(int(1.3 * (n - len(out))) + 16, 4))
        p = p[np.linalg.norm(p[:, :3], axis=1) > C.MIN_RANGE]
        out = np.vstack([out, p])
    return out[:n]


def make_split(n, seed):
    rng = np.random.default_rng(seed)
    pose = sample_poses(n, rng)
    uv, det = A.detect(pose, rng)
    X = A.to_features(uv, det)
    y = D.pose_to_label(pose)
    return X, y, pose

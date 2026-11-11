"""Pose <-> discrete label mapping (4800 classes)."""
import numpy as np
from . import config as C

_EDGES = (C.POSE_MAX - C.POSE_MIN) / C.POSE_BINS


def pose_to_label(pose):
    pose = np.atleast_2d(pose)
    idx = np.floor((pose - C.POSE_MIN) / _EDGES).astype(int)
    idx = np.clip(idx, 0, C.POSE_BINS - 1)
    return np.ravel_multi_index(idx.T, C.POSE_BINS)


def label_to_pose(label):
    """Centre of the label's cell, (N,4)."""
    idx = np.stack(np.unravel_index(np.atleast_1d(label), C.POSE_BINS), axis=1)
    return C.POSE_MIN + (idx + 0.5) * _EDGES


def cell_size():
    return _EDGES.copy()

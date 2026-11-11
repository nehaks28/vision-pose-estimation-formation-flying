"""Run with:  python tests/test_filters.py   (or python -m pytest -q)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseest import aircraft as A, config as C, dynamics as Dy
from poseest.refine import refine


def test_refine_recovers_pose():
    true = np.array([[-55.0, 6.0, 3.0, 12.0]])
    uv, vis = A.project(true)
    est, cost = refine(true + np.array([[4.0, -3.0, 2.0, 5.0]]), uv[0], vis[0].astype(float))
    assert np.abs(est[:3] - true[0, :3]).max() < 0.05 and abs(est[3] - 12.0) < 0.1


def test_trajectory_inside_box_and_range():
    for s in range(10):
        tr = Dy.true_trajectory(s)
        assert (tr >= C.POSE_MIN).all() and (tr <= C.POSE_MAX).all()
        assert np.linalg.norm(tr[:, :3], axis=1).min() > C.MIN_RANGE


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)

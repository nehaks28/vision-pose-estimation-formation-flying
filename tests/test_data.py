"""Run with:  python tests/test_data.py   (or python -m pytest -q)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseest import aircraft as A, discretize as D, config as C


def test_class_count():
    assert C.N_CLASSES == 4800 and A.N_FEAT == 14


def test_label_roundtrip():
    rng = np.random.default_rng(0)
    pose = rng.uniform(C.POSE_MIN, C.POSE_MAX, (500, 4))
    lab = D.pose_to_label(pose)
    centre = D.label_to_pose(lab)
    assert (np.abs(centre - pose) <= D.cell_size() / 2 + 1e-9).all()
    assert (D.pose_to_label(centre) == lab).all()


def test_projection_scales_with_range():
    near, _ = A.project(np.array([[-30.0, 0, 0, 0]]))
    far, _ = A.project(np.array([[-90.0, 0, 0, 0]]))
    assert np.abs(near).max() > 2.5 * np.abs(far).max()


def test_occlusion_changes_with_viewpoint():
    _, behind = A.project(np.array([[-60.0, 0, 0, 0]]))
    _, front = A.project(np.array([[40.0, 0, 0, 0]]))
    assert behind[0, A.FEATURE_NAMES.index("left_nozzle")]
    assert not front[0, A.FEATURE_NAMES.index("left_nozzle")]
    assert front[0, A.FEATURE_NAMES.index("nose")]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)

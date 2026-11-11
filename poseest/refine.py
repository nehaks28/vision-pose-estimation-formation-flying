"""Extension (paper's future work): refine the coarse ANN label with a Gauss-Newton /
Levenberg-Marquardt reprojection-error minimisation over the 4-DoF pose (bundle-adjustment
style, single view). Multiple starts come from the ANN's top-k labels, which are the
neighbouring cells the paper shows misclassifications fall into."""
import numpy as np
from scipy.optimize import least_squares
from . import aircraft as A, config as C

_SCALE = np.array([1.0, 1.0, 1.0, 0.1])      # keeps roll (deg) comparable to metres in the solver


def _resid(q, uv_meas, det):
    uv, _ = A.project(q[None] * _SCALE)
    return ((uv[0] - uv_meas) * det[:, None]).ravel()


def refine(start_poses, uv_meas, det):
    """start_poses: (K,4) candidate poses. Returns (best_pose, best_cost)."""
    best, best_c = None, np.inf
    for s in start_poses:
        r = least_squares(_resid, s / _SCALE, args=(uv_meas, det), method="lm", x_scale=1.0, max_nfev=60)
        c = float((r.fun ** 2).sum())
        if c < best_c:
            best, best_c = r.x * _SCALE, c
    return best, best_c

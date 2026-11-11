"""Relative-motion model and the 10 s formation-approach scenario.

State s = [x, y, z, phi]  (follower position in leader frame, relative roll).
Known to the estimator: the follower's own velocity relative to the leader's
*nominal* motion (`U_KNOWN`).  Unknown (treated as a disturbance, sampled
uniformly by the particle filter, as in the paper): the leader's deviation from
nominal forward speed, its pitch/roll induced vertical/lateral velocity, and its
roll rate.
"""
import numpy as np
from . import config as C

U_KNOWN = np.array([6.0, -0.8, -0.3, 0.0])          # [m/s, m/s, m/s, deg/s]
DIST_BOUND = np.array([2.0, 1.0, 0.8, 5.0])         # |disturbance| bounds (uniform)


def step(s, dist, dt=C.DT):
    """s: (...,4), dist: (...,4) disturbance rates -> next state."""
    return s + dt * (U_KNOWN + dist)


def sample_disturbance(n, rng):
    return rng.uniform(-DIST_BOUND, DIST_BOUND, size=(n, 4))


def true_trajectory(seed=0, horizon=C.HORIZON_S, dt=C.DT):
    """Ground-truth approach: follower closes from ~90 m behind to ~30 m behind while the
    leader gently manoeuvres (smooth disturbances inside the PF's assumed bounds)."""
    rng = np.random.default_rng(seed)
    n = int(round(horizon / dt))
    ph = rng.uniform(0, 2 * np.pi, 4)
    fr = np.array([0.9, 0.7, 1.3, 0.5]) * rng.uniform(0.8, 1.2, 4)
    amp = DIST_BOUND * 0.75
    s = np.array([-90.0 + rng.uniform(-5, 5), rng.uniform(-10, 10), rng.uniform(-6, 6), rng.uniform(-15, 15)])
    traj = [s.copy()]
    for k in range(n - 1):
        t = k * dt
        d = amp * np.sin(fr * t + ph)
        s = step(s, d, dt)
        traj.append(s.copy())
    return np.array(traj)             # (n,4)

"""Aircraft feature model, camera projection and (simplified) self-occlusion.

The leader carries 14 known structural features (nose, canopy, wing tips, tail
tips, engine nozzles, ...). Each feature has a body-frame position and an
outward surface normal; a feature is treated as *occluded* when its normal
faces away from the camera (a cheap stand-in for the paper's ray tracing).
"""
import numpy as np
from . import config as C

# name, body position [x fwd, y right, z up] (m), outward normal
_FEATURES = [
    ("nose",          ( 6.0,  0.0,  0.0), ( 1.0,  0.0,  0.2)),
    ("canopy",        ( 2.5,  0.0,  1.0), ( 0.0,  0.0,  1.0)),
    ("left_wing_tip", (-1.0, -4.5,  0.0), ( 0.0, -1.0,  0.3)),
    ("right_wing_tip",(-1.0,  4.5,  0.0), ( 0.0,  1.0,  0.3)),
    ("left_wing_root",( 0.5, -1.2,  0.0), ( 0.0, -0.3,  1.0)),
    ("right_wing_root",(0.5,  1.2,  0.0), ( 0.0,  0.3,  1.0)),
    ("left_stab_tip", (-5.0, -2.2,  0.5), (-0.3, -1.0,  0.5)),
    ("right_stab_tip",(-5.0,  2.2,  0.5), (-0.3,  1.0,  0.5)),
    ("fin_top",       (-4.5,  0.0,  2.8), (-0.2,  0.0,  1.0)),
    ("left_nozzle",   (-6.0, -0.6,  0.0), (-1.0,  0.0,  0.0)),
    ("right_nozzle",  (-6.0,  0.6,  0.0), (-1.0,  0.0,  0.0)),
    ("belly",         ( 0.0,  0.0, -1.0), ( 0.0,  0.0, -1.0)),
    ("left_intake",   ( 2.0, -0.9, -0.3), ( 0.6, -0.8, -0.3)),
    ("right_intake",  ( 2.0,  0.9, -0.3), ( 0.6,  0.8, -0.3)),
]
N_FEAT = len(_FEATURES)  # 14
FEATURE_NAMES = [f[0] for f in _FEATURES]
BODY_PTS = np.array([f[1] for f in _FEATURES], dtype=np.float64)
_n = np.array([f[2] for f in _FEATURES], dtype=np.float64)
BODY_NRM = _n / np.linalg.norm(_n, axis=1, keepdims=True)


def _rot_x(phi):
    """Rotation about the leader's forward axis. phi: (N,) radians -> (N,3,3)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.zeros(phi.shape + (3, 3))
    R[..., 0, 0] = 1
    R[..., 1, 1] = c; R[..., 1, 2] = -s
    R[..., 2, 1] = s; R[..., 2, 2] = c
    return R


def project(pose):
    """Project the 14 features for a batch of poses.

    pose: (N,4) = [x, y, z (m), phi (deg)].  Returns
      uv  (N,14,2) pixel offsets from the image centre (the gimballed camera
          keeps the leader reference point at the centre),
      vis (N,14) bool, True where the feature faces the camera.
    The follower is at `pose[:3]` in the (level) leader frame, the leader is
    rolled by phi about its own forward axis, and the camera boresight points
    at the leader origin with the horizon level (no camera roll).
    """
    pose = np.atleast_2d(pose).astype(np.float64)
    N = pose.shape[0]
    p = pose[:, :3]
    R = _rot_x(np.deg2rad(pose[:, 3]))                      # (N,3,3)
    pts = np.einsum("nij,kj->nki", R, BODY_PTS)             # body pts in level leader frame
    nrm = np.einsum("nij,kj->nki", R, BODY_NRM)
    d = pts - p[:, None, :]                                 # camera -> point
    fwd = -p / np.linalg.norm(p, axis=1, keepdims=True)     # boresight at leader origin
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right, axis=1, keepdims=True) + 1e-12
    up = np.cross(right, fwd)
    zc = np.maximum(np.einsum("nki,ni->nk", d, fwd), 1e-3)  # depth
    xc = np.einsum("nki,ni->nk", d, right)
    yc = np.einsum("nki,ni->nk", d, up)
    uv = np.stack([C.FOCAL_PX * xc / zc, C.FOCAL_PX * yc / zc], axis=-1)
    vis = np.einsum("nki,nki->nk", nrm, -d) > 0.0           # normal faces camera
    return uv, vis


def to_features(uv, detected):
    """Network input: [u/(W/2), v/(H/2), occluded] x 14 -> 42 floats (occluded => u=v=0)."""
    N = uv.shape[0]
    f = np.zeros((N, N_FEAT, 3), dtype=np.float32)
    f[..., 0] = np.where(detected, uv[..., 0] / (C.IMG_W / 2), 0.0)
    f[..., 1] = np.where(detected, uv[..., 1] / (C.IMG_H / 2), 0.0)
    f[..., 2] = (~detected).astype(np.float32)
    return f.reshape(N, N_FEAT * 3)


def detect(pose, rng):
    """Simulated CNN feature detector: visible features + Gaussian pixel noise +
    random misses. Returns noisy uv (N,14,2) and detected mask (N,14)."""
    uv, vis = project(pose)
    det = vis & (rng.random(vis.shape) > C.DROPOUT_PROB)
    uv = uv + rng.normal(0.0, C.PIXEL_NOISE_STD, uv.shape)
    return uv, det

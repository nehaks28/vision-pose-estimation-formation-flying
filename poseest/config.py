"""Shared constants: camera, pose discretisation, simulation settings."""
import numpy as np

# ---- Camera (pinhole, gimballed to always look at the leader) -------------
IMG_W, IMG_H = 1280, 960
FOCAL_PX = 1600.0            # focal length in pixels
PIXEL_NOISE_STD = 0.5        # std-dev of feature detector noise (px)
DROPOUT_PROB = 0.02          # prob. that a visible feature is missed by the detector
MIN_RANGE = 15.0             # m, never closer than this (leader is ~12 m long)

# ---- Pose discretisation (Table I of the paper) ---------------------------
# pose = [x fwd (m), y lateral (m), z vertical (m), phi roll about line of sight (deg)]
# (follower position expressed in the leader frame; negative x = follower behind leader)
POSE_MIN = np.array([-100.0, -50.0, -20.0, -45.0])
POSE_MAX = np.array([50.0, 50.0, 20.0, 45.0])
POSE_BINS = np.array([10, 10, 8, 6])
N_CLASSES = int(np.prod(POSE_BINS))        # 4800
assert N_CLASSES == 4800

# ---- Dynamics -------------------------------------------------------------
DT = 0.1                      # s (10 Hz camera)
HORIZON_S = 10.0              # s, length of the approach scenario

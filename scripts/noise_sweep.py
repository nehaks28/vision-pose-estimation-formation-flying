"""Robustness: how do the estimators degrade as detector pixel noise grows?
(the ANN is *not* retrained - it was trained at 0.5 px noise).

    python scripts/noise_sweep.py --seeds 5
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from poseest import dynamics as Dy, config as C, pipeline as P
from poseest.ann import PoseMLP

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
net = PoseMLP.load(f"{ROOT}/artifacts/ann_weights.npz"); pool = np.load(f"{ROOT}/artifacts/error_pool.npy")
levels = [0.5, 1.0, 2.0, 4.0, 8.0]
out = {}
for sd in levels:
    C.PIXEL_NOISE_STD = sd
    acc = {}
    for s in range(a.seeds):
        traj = Dy.true_trajectory(seed=s)
        est, _ = P.run_all(net, pool, traj, seed=s)
        for k, e in est.items():
            acc.setdefault(k, []).append(P.errors(e, traj)[0].mean())
    out[sd] = {k: float(np.mean(v)) for k, v in acc.items()}
    print(sd, {k[:22]: round(v, 3) for k, v in out[sd].items()}, flush=True)
json.dump(out, open(f"{ROOT}/results/noise_sweep.json", "w"), indent=2)
cols = ["#888888", "#e08a1e", "#c9501c", "#2a7ab9", "#1b9e5a"]
short = ["ANN only", "PF cfg1", "PF cfg2", "PF Gaussian", "ANN+GN (ours)"]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for i, k in enumerate(out[levels[0]]):
    ax.plot(levels, [out[l][k] for l in levels], "o-", color=cols[i], label=short[i], ms=4)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("detector pixel noise std (px)"); ax.set_ylabel("mean position error (m)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{ROOT}/results/noise_sweep.png", dpi=170)

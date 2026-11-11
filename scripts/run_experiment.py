"""Evaluate all estimators on several random 10 s formation-approach trajectories.

    python scripts/run_experiment.py --seeds 5
Writes results/results.json and results/*.png
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from poseest import dynamics as Dy, config as C, aircraft as A, pipeline as P
from poseest.ann import PoseMLP

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

net = PoseMLP.load(f"{ROOT}/artifacts/ann_weights.npz")
pool = np.load(f"{ROOT}/artifacts/error_pool.npy")
names = None
per_seed_pos, per_seed_roll, series = {}, {}, {}
for s in range(a.seeds):
    traj = Dy.true_trajectory(seed=s)
    est, _ = P.run_all(net, pool, traj, seed=s)
    for k, e in est.items():
        pe, re = P.errors(e, traj)
        per_seed_pos.setdefault(k, []).append(pe.mean())
        per_seed_roll.setdefault(k, []).append(re.mean())
        series.setdefault(k, []).append((pe, re))
    print(f"seed {s} done:", {k: round(float(v[-1]), 3) for k, v in per_seed_pos.items()}, flush=True)

res = {k: dict(pos_mean_m=float(np.mean(v)), pos_std_m=float(np.std(v)),
               roll_mean_deg=float(np.mean(per_seed_roll[k])), roll_std_deg=float(np.std(per_seed_roll[k])))
       for k, v in per_seed_pos.items()}
json.dump(res, open(f"{ROOT}/results/results.json", "w"), indent=2)
print(f"\n{'method':58s} {'pos err (m)':>16s} {'roll err (deg)':>16s}")
for k, r in res.items():
    print(f"{k:58s} {r['pos_mean_m']:7.3f} ± {r['pos_std_m']:5.3f}   {r['roll_mean_deg']:7.3f} ± {r['roll_std_deg']:5.3f}")

# ---------------- figures ----------------
cols = ["#888888", "#e08a1e", "#c9501c", "#2a7ab9", "#1b9e5a"]
t = np.arange(len(series["ANN only"][0][0])) * C.DT
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for (k, v), c in zip(series.items(), cols):
    ax[0].plot(t, np.mean([x[0] for x in v], 0), label=k, color=c, lw=1.6)
    ax[1].plot(t, np.mean([x[1] for x in v], 0), color=c, lw=1.6)
ax[0].set_ylabel("position error (m)"); ax[1].set_ylabel("roll error (deg)")
for x in ax: x.set_xlabel("time (s)"); x.grid(alpha=.3)
ax[0].set_yscale("log"); ax[1].set_yscale("log")
fig.legend(loc="lower center", ncol=2, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(f"{ROOT}/results/error_vs_time.png", dpi=170)

fig, ax = plt.subplots(figsize=(4.6, 3.3))
short = ["ANN only", "PF cfg1", "PF cfg2", "PF Gauss", "ANN+GN"]
vals = [res[k]["pos_mean_m"] for k in res]; errs = [res[k]["pos_std_m"] for k in res]
ax.bar(short, vals, yerr=errs, color=cols, capsize=3)
ax.set_yscale("log"); ax.set_ylabel("mean position error (m)"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{ROOT}/results/position_error_bar.png", dpi=170)
print("figures written")

"""Live demo: simulate one 10 s formation approach, run every estimator, show/save the result.

    python scripts/demo.py --seed 3            # saves results/demo.gif and prints a summary table
    python scripts/demo.py --seed 3 --live     # also opens an interactive matplotlib window
"""
import argparse, os, sys, time
import numpy as np
import matplotlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--live", action="store_true")
ap.add_argument("--no-gif", action="store_true")
a = ap.parse_args()
if not a.live:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from poseest import dynamics as Dy, config as C, aircraft as A, pipeline as P
from poseest.ann import PoseMLP

net = PoseMLP.load(f"{ROOT}/artifacts/ann_weights.npz")
pool = np.load(f"{ROOT}/artifacts/error_pool.npy")
traj = Dy.true_trajectory(a.seed)
t0 = time.time()
est, obs = P.run_all(net, pool, traj, seed=a.seed)
print(f"processed {len(traj)} frames with {len(est)} estimators in {time.time() - t0:.1f}s\n")
print(f"{'method':58s} {'pos err (m)':>12s} {'roll err (deg)':>15s}")
for k, e in est.items():
    pe, re = P.errors(e, traj)
    print(f"{k:58s} {pe.mean():12.3f} {re.mean():15.3f}")

# ---------------- visualisation ----------------
names = list(est)
sty = {"ANN only": ("#888888", "--"), names[1]: ("#e08a1e", "-"), names[2]: ("#c9501c", "-"),
       names[3]: ("#2a7ab9", "-"), names[4]: ("#1b9e5a", "-")}
short = {names[0]: "ANN only", names[1]: "PF cfg1", names[2]: "PF cfg2", names[3]: "PF Gaussian", names[4]: "ANN+GN (ours)"}
t = np.arange(len(traj)) * C.DT
fig = plt.figure(figsize=(11, 4.6))
ax_img = fig.add_subplot(1, 3, 1); ax_x = fig.add_subplot(1, 3, 2); ax_e = fig.add_subplot(1, 3, 3)
LINKS = [(0, 2), (0, 3), (2, 6), (3, 7), (6, 9), (7, 10), (8, 9), (8, 10), (4, 5)]


def draw(i):
    ax_img.clear(); ax_x.clear(); ax_e.clear()
    uv, det = obs["uv"][i], obs["det"][i]
    ax_img.set_facecolor("#cfe3f4"); ax_img.set_xlim(-300, 300); ax_img.set_ylim(-225, 225); ax_img.set_aspect("equal")
    full, _ = A.project(traj[i:i + 1])
    for a_, b_ in LINKS:
        ax_img.plot(full[0, [a_, b_], 0], full[0, [a_, b_], 1], color="#7a8a99", lw=1)
    ax_img.scatter(*uv[det].T, c="red", s=26, zorder=3, label="detected")
    ax_img.scatter(*full[0][~det].T, facecolors="none", edgecolors="#555", s=26, label="occluded")
    ax_img.set_title(f"follower camera  (t = {t[i]:.1f}s)", fontsize=9); ax_img.legend(fontsize=7, loc="upper right")
    ax_x.plot(t, traj[:, 0], "k", lw=2.2, label="truth")
    for k, e in est.items():
        ax_x.plot(t[:i + 1], e[:i + 1, 0], color=sty[k][0], ls=sty[k][1], lw=1.4, label=short[k])
    ax_x.set_xlim(0, t[-1]); ax_x.set_title("forward position x (m)", fontsize=9); ax_x.grid(alpha=.3); ax_x.legend(fontsize=6.5)
    for k, e in est.items():
        pe, _ = P.errors(e, traj)
        ax_e.plot(t[:i + 1], pe[:i + 1], color=sty[k][0], ls=sty[k][1], lw=1.4, label=short[k])
    ax_e.set_yscale("log"); ax_e.set_ylim(1e-2, 20); ax_e.set_xlim(0, t[-1]); ax_e.grid(alpha=.3)
    ax_e.set_title("position error (m)", fontsize=9)
    fig.tight_layout()


if a.live:
    plt.ion()
    for i in range(len(traj)):
        draw(i); plt.pause(0.05)
    plt.ioff(); plt.show()
if not a.no_gif:
    from matplotlib.animation import FuncAnimation, PillowWriter
    FuncAnimation(fig, draw, frames=range(0, len(traj), 2), interval=100).save(f"{ROOT}/results/demo.gif", writer=PillowWriter(fps=10))
    draw(len(traj) - 1); fig.savefig(f"{ROOT}/results/demo_final_frame.png", dpi=160)
    print("\nsaved results/demo.gif and results/demo_final_frame.png")

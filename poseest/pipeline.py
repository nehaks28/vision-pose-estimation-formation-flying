"""End-to-end estimators evaluated on a trajectory."""
import numpy as np
from . import aircraft as A, discretize as D, dynamics as Dy, config as C
from .particle_filter import ANNInformedPF
from .refine import refine

PF_CONFIGS = {
    # name: kwargs for ANNInformedPF
    "ANN + PF (paper cfg 1: N=1000, a=0.9 / 10 cycles)": dict(n_particles=1000, alpha=0.9, alpha_every=10, likelihood="inv_mse"),
    "ANN + PF (paper cfg 2: N=5000, a=0.9 every cycle)": dict(n_particles=5000, alpha=0.9, alpha_every=1, likelihood="inv_mse"),
    "ANN + PF (Gaussian lik., N=2000, a=0.5)": dict(n_particles=2000, alpha=0.5, alpha_every=1, likelihood="gauss",
                                                     sigma_px=6.0, jitter=(0.3, 0.3, 0.15, 0.3)),
}


def observe_trajectory(traj, rng):
    """Noisy detected features for each frame."""
    return A.detect(traj, rng)                       # uv (T,14,2), det (T,14)


def run_all(net, error_pool, traj, seed, pf_configs=PF_CONFIGS, topk=3):
    rng = np.random.default_rng(1000 + seed)
    uv, det = observe_trajectory(traj, rng)
    X = A.to_features(uv, det)
    top = net.topk(X, topk)                          # (T,k) labels
    label_pose = D.label_to_pose(top[:, 0])
    est = {"ANN only": label_pose.copy()}
    for name, kw in pf_configs.items():
        pf = ANNInformedPF(error_pool, seed=seed, **kw)
        est[name] = np.array([pf.step(label_pose[t], uv[t], det[t]) for t in range(len(traj))])
    gn = []
    for t in range(len(traj)):
        starts = D.label_to_pose(top[t])
        gn.append(refine(starts, uv[t], det[t].astype(float))[0])
    est["ANN + Gauss-Newton refinement (ours)"] = np.array(gn)
    return est, dict(uv=uv, det=det, top1=top[:, 0])


def errors(est, traj):
    d = est - traj
    return np.linalg.norm(d[:, :3], axis=1), np.abs(d[:, 3])

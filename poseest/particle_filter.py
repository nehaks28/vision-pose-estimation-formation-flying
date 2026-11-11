"""Particle filter that is *informed by the ANN pose label* (Sec. IV-E of the paper).

Per time step:
  1. propagate particles through the relative dynamics with uniformly sampled leader disturbance
  2. project each particle's 14 features and weight it against the detected features
     (only features that were actually detected are used; particle self-occlusion is ignored)
  3. resample: floor(alpha*N) from the weighted set, the rest from  ANN label + error-pool delta.

Two likelihoods are provided:
  'inv_mse'  : w = 1 / (MSE + eps)        -- the paper's likelihood
  'gauss'    : w = exp(-0.5 * SSE/sigma^2) -- sharper, with temperature `temper` to avoid collapse
"""
import numpy as np
from . import aircraft as A, config as C, dynamics as Dy, discretize as D


def _residual_sq(particles, uv_meas, det):
    uv_p, _ = A.project(particles)                      # (N,14,2)
    r2 = ((uv_p - uv_meas[None]) ** 2).sum(-1)          # (N,14)
    r2 = r2 * det[None]
    n_det = max(det.sum(), 1)
    return r2.sum(1), n_det


class ANNInformedPF:
    def __init__(self, error_pool, n_particles=1000, alpha=0.9, alpha_every=1,
                 likelihood="inv_mse", sigma_px=3.0, temper=1.0, jitter=(0.05, 0.05, 0.03, 0.05),
                 seed=0):
        self.pool = error_pool
        self.N = n_particles
        self.alpha, self.alpha_every = alpha, alpha_every
        self.lik, self.sigma, self.temper = likelihood, sigma_px, temper
        self.jitter = np.array(jitter)
        self.rng = np.random.default_rng(seed)
        self.k = 0
        self.p = None
        self.w = None

    def _sample_from_label(self, label_pose, n):
        d = self.pool[self.rng.integers(0, len(self.pool), n)]
        return label_pose[None] + d

    def _weights(self, uv_meas, det):
        sse, n_det = _residual_sq(self.p, uv_meas, det)
        if self.lik == "inv_mse":
            w = 1.0 / (sse / n_det + 1e-6)
        else:
            ll = -0.5 * sse / (self.sigma ** 2) / self.temper
            w = np.exp(ll - ll.max())
        return w / w.sum()

    def step(self, label_pose, uv_meas, det):
        """label_pose: (4,) centre of ANN's predicted cell for this frame."""
        if self.p is None:
            self.p = self._sample_from_label(label_pose, self.N)
        else:
            self.p = Dy.step(self.p, Dy.sample_disturbance(self.N, self.rng))
        self.w = self._weights(uv_meas, det)
        est = (self.w[:, None] * self.p).sum(0)
        # --- resample ---
        a = self.alpha if (self.k % self.alpha_every == 0) else 1.0
        n_w = int(np.floor(a * self.N))
        idx = self.rng.choice(self.N, size=n_w, p=self.w)
        new = self.p[idx]
        if n_w < self.N:
            new = np.vstack([new, self._sample_from_label(label_pose, self.N - n_w)])
        self.p = new + self.rng.normal(0, 1, new.shape) * self.jitter   # roughening
        self.k += 1
        return est

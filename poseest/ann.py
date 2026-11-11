"""Pose-label classifier: 42 -> 100 -> 100 -> 4800 (ReLU, softmax), Adam.

Pure numpy (float32) so the project has no deep-learning-framework dependency.
Architecture follows the paper exactly (Sec. IV-D).
"""
import time
import numpy as np
from . import config as C


class PoseMLP:
    def __init__(self, n_in=42, hidden=(100, 100), n_out=C.N_CLASSES, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mu = np.zeros(n_in, np.float32)
        self.sd = np.ones(n_in, np.float32)

    # ---- inference -------------------------------------------------------
    def logits(self, X):
        h = (X - self.mu) / self.sd
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return h @ self.W[-1] + self.b[-1]

    def predict_proba(self, X):
        z = self.logits(X)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X, bs=4096):
        return np.concatenate([self.logits(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])

    def topk(self, X, k=3):
        z = self.logits(X)
        idx = np.argpartition(-z, k, axis=1)[:, :k]
        order = np.take_along_axis(z, idx, 1).argsort(1)[:, ::-1]
        return np.take_along_axis(idx, order, 1)

    # ---- training --------------------------------------------------------
    def fit(self, X, y, Xv, yv, epochs=30, bs=256, lr=2e-3, lr_end=2e-4, log=print):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        rng = np.random.default_rng(1)
        n = len(Xs)
        hist = []
        for ep in range(epochs):
            t0 = time.time()
            cur_lr = lr * (lr_end / lr) ** (ep / max(1, epochs - 1))   # exponential decay
            perm = rng.permutation(n)
            tot = 0.0
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]
                xb, yb = Xs[idx], y[idx]
                # forward
                a0 = xb
                z1 = a0 @ self.W[0] + self.b[0]; a1 = np.maximum(z1, 0)
                z2 = a1 @ self.W[1] + self.b[1]; a2 = np.maximum(z2, 0)
                z3 = a2 @ self.W[2] + self.b[2]
                z3 -= z3.max(1, keepdims=True)
                np.maximum(z3, -30.0, out=z3)        # avoid float32 denormals (huge slowdown)
                p = np.exp(z3); p /= p.sum(1, keepdims=True)
                tot += -np.log(p[np.arange(bs), yb] + 1e-12).mean()
                # backward (softmax cross-entropy)
                d3 = p; d3[np.arange(bs), yb] -= 1; d3 /= bs
                gW3 = a2.T @ d3; gb3 = d3.sum(0)
                d2 = (d3 @ self.W[2].T) * (z2 > 0)
                gW2 = a1.T @ d2; gb2 = d2.sum(0)
                d1 = (d2 @ self.W[1].T) * (z1 > 0)
                gW1 = a0.T @ d1; gb1 = d1.sum(0)
                grads = [gW1, gW2, gW3, gb1, gb2, gb3]
                t += 1
                for j, g in enumerate(grads):                                # Adam
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mh = m[j] / (1 - b1 ** t); vh = v[j] / (1 - b2 ** t)
                    params[j] -= (cur_lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            acc = (self.predict(Xv) == yv).mean()
            hist.append((ep + 1, tot / (n // bs), acc))
            log(f"epoch {ep + 1:2d}/{epochs}  loss {tot / (n // bs):.3f}  val-acc {acc:.3f}  ({time.time() - t0:.0f}s)")
        return hist

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, mu=self.mu, sd=self.sd,
                            **{f"W{i}": w for i, w in enumerate(self.W)},
                            **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        net = cls()
        net.mu, net.sd = z["mu"], z["sd"]
        net.W = [z[f"W{i}"] for i in range(3)]
        net.b = [z[f"b{i}"] for i in range(3)]
        return net

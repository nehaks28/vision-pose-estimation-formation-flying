"""Generate synthetic data, train the pose-label ANN, save weights + error pool.

    python scripts/train_ann.py --data-dir ../pose-dataset --epochs 30
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poseest import dataset, data_io, discretize as D, config as C
from poseest.ann import PoseMLP

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=300_000)
ap.add_argument("--n-val", type=int, default=30_000)
ap.add_argument("--n-test", type=int, default=30_000)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--out", default="artifacts")
ap.add_argument("--data-dir", default=None,
                help="folder with train.csv/val.csv/test.csv (the separate pose-dataset folder). "
                     "If omitted, the data is regenerated in memory with the same seeds.")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if a.data_dir:
    print(f"loading dataset CSVs from {a.data_dir} ...")
    Xtr, ytr, _ = data_io.load_split(a.data_dir, "train")
    Xva, yva, pva = data_io.load_split(a.data_dir, "val")
    Xte, yte, pte = data_io.load_split(a.data_dir, "test")
else:
    Xtr, ytr, _ = dataset.make_split(a.n_train, seed=11)
    Xva, yva, pva = dataset.make_split(a.n_val, seed=22)
    Xte, yte, pte = dataset.make_split(a.n_test, seed=33)
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}; classes used in train: {len(np.unique(ytr))}/{C.N_CLASSES}")

net = PoseMLP()
hist = net.fit(Xtr, ytr, Xva, yva, epochs=a.epochs)
net.save(f"{a.out}/ann_weights.npz")

# ---- error pool (true pose - classified pose) from the validation set, used by the PF ----
pred_va = net.predict(Xva)
np.save(f"{a.out}/error_pool.npy", (pva - D.label_to_pose(pred_va)).astype(np.float32))

# ---- test metrics ----
top5 = net.topk(Xte, 5)
pred = top5[:, 0]
cell = D.cell_size()
true_idx = np.stack(np.unravel_index(yte, C.POSE_BINS), 1)
pred_idx = np.stack(np.unravel_index(pred, C.POSE_BINS), 1)
adj = (np.abs(true_idx - pred_idx).max(1) <= 1)
err = pte - D.label_to_pose(pred)
metrics = dict(
    test_top1=float((pred == yte).mean()),
    test_top5=float((top5 == yte[:, None]).any(1).mean()),
    test_within_one_cell=float(adj.mean()),
    test_pos_err_mean_m=float(np.linalg.norm(err[:, :3], axis=1).mean()),
    test_pos_err_median_m=float(np.median(np.linalg.norm(err[:, :3], axis=1))),
    test_roll_err_mean_deg=float(np.abs(err[:, 3]).mean()),
    cell_size=cell.tolist(),
    history=[dict(epoch=int(e), loss=float(l), val_acc=float(v)) for e, l, v in hist],
)
json.dump(metrics, open(f"{a.out}/ann_metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

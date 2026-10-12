"""Recompute order-free metrics for every method in results.json from the saved test predictions:
MCC, balanced accuracy, Cohen's kappa, top-3 accuracy (methods that store class probabilities only), and
tail recall (mean recall over the third of classes with the fewest training examples).

Used for methods scored before a metric existed; the 512-token ablation has no saved predictions.
"""
import gzip
import json

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, recall_score

import store
from evaluate import PRED_DIR


def merge(res, load):
    for name, entry in res["datasets"].items():
        path = PRED_DIR / f"{name}.json.gz"
        if not path.exists():
            continue
        p = json.loads(gzip.decompress(path.read_bytes()))
        y = np.array(p["y"])
        n = len(entry["meta"]["labels"])
        train_counts = np.array(entry["meta"].get("counts", {}).get("train") or entry["meta"].get("imbalance", {}).get("train_counts") or [])
        tail = np.argsort(train_counts)[: max(1, n // 3)] if len(train_counts) == n else None
        for method, pred in p["pred"].items():
            m = entry["methods"].get(method)
            if m is None:
                continue
            m["mcc"] = float(matthews_corrcoef(y, pred))
            m["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
            m["kappa"] = float(cohen_kappa_score(y, pred))
            if tail is not None:
                rec = recall_score(y, pred, labels=range(n), average=None, zero_division=0)
                m["tail_recall"] = float(rec[tail].mean())
            if method in p.get("top3", {}):
                m["top3_accuracy"] = float((np.array(p["top3"][method]) == y[:, None]).any(1).mean())
        print(name, {k: round(v.get("top3_accuracy", float("nan")), 3) for k, v in entry["methods"].items()}, flush=True)


if __name__ == "__main__":
    store.update(merge)

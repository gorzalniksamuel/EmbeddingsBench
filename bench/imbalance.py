"""Class-imbalance schemes on frozen embeddings, compared with the plain classifiers.

  logreg_balanced  LogReg with class weights ∝ 1 / class frequency
  logit_adjusted   plain LogReg, post-hoc: argmax(log p − τ·log prior), τ chosen on validation (Menon et al., 2021)
  mlp_weighted     MLP (benchmark architecture) with class-weighted cross-entropy
  mlp_balanced     same MLP, trained on class-balanced batches (sampling ∝ 1 / class frequency)

Settings are chosen on validation macro-F1 (accuracy would reward predicting the head classes). Validation and
test keep the natural class distribution. Adds "tail_recall" (mean recall over the third of classes with the
fewest training examples) to every method of the dataset.

usage: python imbalance.py [--force] <dataset> [...]     (existing results are kept unless --force)
"""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import mlp_search
import store
from data import DATA_DIR
from embed import EMB_DIR, MAX_TOKENS
from evaluate import DEVICE, RESULTS, metrics

C_GRID = (0.1, 1.0, 10.0, 100.0)
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def tail_classes(train_counts):
    order = np.argsort(train_counts)
    return order[: max(1, len(order) // 3)]


def tail_recall(per_class_recall, tail):
    return float(np.mean([per_class_recall[i] for i in tail]))


def fit_mlp(p, x, y, xv, yv, n, weights=None, balanced=False, seed=13, max_epochs=300, patience=10):
    """mlp_search.fit with class-weighted loss or class-balanced batches; early stopping on validation loss."""
    torch.manual_seed(seed)
    model = mlp_search.build(x.shape[1], n, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    xt, yt = torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)
    xvt, yvt = torch.tensor(xv, device=DEVICE), torch.tensor(yv, device=DEVICE)
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE) if weights is not None else None
    batch = min(256, max(8, len(x) // 4))
    gen = torch.Generator().manual_seed(seed)
    counts = np.bincount(y, minlength=n)
    sample_w = torch.tensor(1.0 / counts[y], dtype=torch.float64)
    best, state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        idx_all = torch.multinomial(sample_w, len(y), replacement=True, generator=gen) if balanced else torch.randperm(len(y), generator=gen)
        for idx in idx_all.split(batch):
            idx = idx.to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.cross_entropy(model(xvt), yvt, weight=w).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


SCHEMES = ("logreg_balanced", "logit_adjusted", "mlp_weighted", "mlp_balanced")


def run(name, results):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    entry = results["datasets"][name]
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    emb = np.load(EMB_DIR / f"{name}.npz")
    x, xv, xt = (emb[f"{s}_{MAX_TOKENS}"] for s in ("train", "val", "test"))
    counts = np.bincount(y, minlength=n)
    prior = counts / counts.sum()
    tail = tail_classes(counts)
    out, preds = {}, {}

    # class-weighted LogReg
    table, best = [], None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(x, y)
        f = macro_f1(yv, clf.predict(xv))
        table.append({"params": {"C": C}, "val_macro_f1": f, "val_acc": float((clf.predict(xv) == yv).mean())})
        print(f"  logreg_balanced C={C}: val macro-F1 {f:.4f}", flush=True)
        if best is None or f > best[1]:
            best = (clf, f, {"C": C})
    proba_bal = best[0].predict_proba(xt)
    preds["logreg_balanced"] = proba_bal
    out["logreg_balanced"] = {"params": best[2], "val_macro_f1": best[1], "selection": table,
                              "val_acc": next(t["val_acc"] for t in table if t["params"] == best[2])}

    # post-hoc logit adjustment of the plain LogReg (benchmark C)
    C = entry["methods"]["logreg"]["params"]["C"]
    plain = LogisticRegression(C=C, max_iter=1000).fit(x, y)
    lv, lt = np.log(plain.predict_proba(xv) + 1e-12), np.log(plain.predict_proba(xt) + 1e-12)
    adj = lambda logp, tau: logp - tau * np.log(prior + 1e-12)
    table = [{"params": {"tau": t}, "val_macro_f1": macro_f1(yv, adj(lv, t).argmax(1)), "val_acc": float((adj(lv, t).argmax(1) == yv).mean())} for t in TAUS]
    top = max(table, key=lambda r: r["val_macro_f1"])
    s = adj(lt, top["params"]["tau"])
    preds["logit_adjusted"] = np.exp(s - s.max(1, keepdims=True)) / np.exp(s - s.max(1, keepdims=True)).sum(1, keepdims=True)
    out["logit_adjusted"] = {"params": {**top["params"], "C": C}, "val_macro_f1": top["val_macro_f1"], "selection": table,
                             "val_acc": top["val_acc"]}
    print(f"  logit_adjusted tau={top['params']['tau']}: val macro-F1 {top['val_macro_f1']:.4f}", flush=True)

    # MLPs with the benchmark's tuned architecture (or the default) and imbalance-aware training
    arch = entry["methods"].get("mlp_tuned", {}).get("params") or mlp_search.MANUAL_DEFAULTS
    weights = counts.sum() / (n * np.maximum(counts, 1))
    for key, kw in (("mlp_weighted", {"weights": weights}), ("mlp_balanced", {"balanced": True})):
        model = fit_mlp(arch, x, y, xv, yv, n, **kw)
        pv, pt = mlp_search.proba(model, xv), mlp_search.proba(model, xt)
        preds[key] = pt
        out[key] = {"params": arch, "val_macro_f1": macro_f1(yv, pv.argmax(1)), "val_acc": float((pv.argmax(1) == yv).mean()),
                    "selection": [{"params": arch, "val_acc": float((pv.argmax(1) == yv).mean())}]}
        print(f"  {key}: val macro-F1 {out[key]['val_macro_f1']:.4f}", flush=True)

    for key, pt in preds.items():
        pred = pt.argmax(1)
        out[key] = {**metrics(yt, pred, n), **out[key], "selected_on": "validation macro-F1"}
        out[key]["tail_recall"] = tail_recall(out[key]["per_class"]["recall"], tail)
        print(f"  {key:16s} test acc {out[key]['accuracy']:.4f} macro-F1 {out[key]['macro_f1']:.4f} "
              f"bal.acc {out[key]['balanced_accuracy']:.4f} tail recall {out[key]['tail_recall']:.4f}", flush=True)

    def merge(res, load):
        ds = res["datasets"][name]
        for key, r in out.items():
            ds["methods"][key] = r
        for key, m in ds["methods"].items():  # tail recall for every method
            m["tail_recall"] = tail_recall(m["per_class"]["recall"], tail)
        ds["meta"]["imbalance"] = {"ratio": float(counts.max() / max(counts.min(), 1)), "majority_share": float(counts.max() / counts.sum()),
                                   "tail_classes": [labels[i] for i in tail], "train_counts": counts.tolist()}
        p = load(name)
        for key, pt in preds.items():
            top3 = np.argsort(-pt, axis=1)[:, :3]
            p["pred"][key] = pt.argmax(1).tolist()
            p["top3"][key] = top3.tolist()
            p["top3p"][key] = np.take_along_axis(pt, top3, 1).round(4).tolist()
    store.update(merge)


def main():
    force = "--force" in sys.argv
    results = json.loads(RESULTS.read_text())
    for name in [a for a in sys.argv[1:] if a != "--force"]:
        if not force and all(k in results["datasets"][name]["methods"] for k in SCHEMES):
            print(name, "already has the imbalance schemes (use --force to recompute)", flush=True)
            continue
        print(name, flush=True)
        run(name, results)


if __name__ == "__main__":
    main()

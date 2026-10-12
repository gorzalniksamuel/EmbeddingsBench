"""Extra classifiers on the cached embeddings, same protocol as evaluate.py (fit on train, select on validation,
score test once), merged into results.json and the prediction files:

- svm_rbf:   kernel SVM (RBF) on the frozen embeddings.
- ensemble:  weighted average of embedding-LogReg probabilities and temperature-scaled TF-IDF SVM scores.
- matryoshka: LogReg test accuracy when the 1,024-d embeddings are cut to their first d dimensions.
"""
import json
import time

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

import store
from data import DATA_DIR, SPECS
from embed import EMB_DIR, MAX_TOKENS
from evaluate import RESULTS, metrics

SVM_C = (1.0, 10.0, 100.0)
WEIGHTS = tuple(round(w, 1) for w in np.linspace(0, 1, 11))
TEMPERATURES = (1.0, 3.0, 10.0, 30.0)
DIMS = (32, 64, 128, 256, 512, 1024)
SVM_MAX_TRAIN = 30_000  # the RBF SVM scales ~N²; larger training sets are subsampled (stratified) for it


def run(name, chosen):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    emb = np.load(EMB_DIR / f"{name}.npz")
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    x, xv, xt = (emb[f"{s}_{MAX_TOKENS}"] for s in ("train", "val", "test"))
    out = {}

    # RBF SVM (on a stratified subsample when the training set is large)
    sx, sy, sub = x, y, None
    if len(y) > SVM_MAX_TRAIN:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(y)), train_size=SVM_MAX_TRAIN, stratify=y, random_state=13)
        sx, sy, sub = x[idx], y[idx], len(idx)
        print(f"  svm_rbf: fitting on a stratified subsample of {sub:,} of {len(y):,} training texts", flush=True)
    table, best = [], None
    for C in SVM_C:
        t0 = time.time()
        clf = SVC(C=C, kernel="rbf", gamma="scale").fit(sx, sy)
        acc = accuracy_score(yv, clf.predict(xv))
        table.append({"params": {"C": C}, "val_acc": acc})
        print(f"  svm_rbf C={C}: val {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or acc > best[1]:
            best = (clf, acc, {"C": C})
    pred_svm = best[0].predict(xt)
    out["svm_rbf"] = {**metrics(yt, pred_svm, n), "params": best[2], "val_acc": best[1], "selection": table,
                      **({"train_subsample": sub} if sub else {})}
    print(f"  svm_rbf test {out['svm_rbf']['accuracy']:.4f}", flush=True)

    # ensemble: embeddings LogReg + TF-IDF linear SVM
    lr = LogisticRegression(C=chosen["logreg"]["C"], max_iter=1000).fit(x, y)
    pv_emb, pt_emb = lr.predict_proba(xv), lr.predict_proba(xt)
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    svm = LinearSVC(C=chosen["tfidf_svm"]["C"]).fit(vec.fit_transform(texts["train"]), y)
    dv, dt = svm.decision_function(vec.transform(texts["val"])), svm.decision_function(vec.transform(texts["test"]))
    grid = []
    for w in WEIGHTS:
        for T in TEMPERATURES:
            p = w * pv_emb + (1 - w) * softmax(T * dv, axis=1)
            grid.append({"params": {"w": w, "T": T}, "val_acc": accuracy_score(yv, p.argmax(1))})
    top = max(grid, key=lambda g: g["val_acc"])
    w, T = top["params"]["w"], top["params"]["T"]
    proba = w * pt_emb + (1 - w) * softmax(T * dt, axis=1)
    pred_ens = proba.argmax(1)
    per_w = [max((g for g in grid if g["params"]["w"] == wv), key=lambda g: g["val_acc"]) for wv in WEIGHTS]
    out["ensemble"] = {**metrics(yt, pred_ens, n), "params": top["params"], "val_acc": top["val_acc"], "selection": per_w}
    print(f"  ensemble w={w} T={T}: val {top['val_acc']:.4f} test {out['ensemble']['accuracy']:.4f}", flush=True)

    # Matryoshka truncation
    mat = []
    for dim in DIMS:
        cut = lambda a: a[:, :dim] / np.linalg.norm(a[:, :dim], axis=1, keepdims=True)
        clf = LogisticRegression(C=chosen["logreg"]["C"], max_iter=1000).fit(cut(x), y)
        mat.append({"dims": dim, "accuracy": accuracy_score(yt, clf.predict(cut(xt)))})
        print(f"  matryoshka {dim}: {mat[-1]['accuracy']:.4f}", flush=True)

    top3 = np.argsort(-proba, axis=1)[:, :3]

    def merge(res, load):
        ds = res["datasets"][name]
        ds["methods"]["svm_rbf"] = out["svm_rbf"]
        ds["methods"]["ensemble"] = out["ensemble"]
        ds["matryoshka"] = mat
        p = load(name)
        p["pred"]["svm_rbf"] = pred_svm.tolist()
        p["pred"]["ensemble"] = pred_ens.tolist()
        p["top3"]["ensemble"] = top3.tolist()
        p["top3p"]["ensemble"] = np.take_along_axis(proba, top3, 1).round(4).tolist()
    store.update(merge)


def main():
    import sys
    results = json.loads(RESULTS.read_text())
    for name in sys.argv[1:] or SPECS:
        print(name, flush=True)
        run(name, {m: results["datasets"][name]["methods"][m]["params"] for m in ("logreg", "tfidf_svm")})


if __name__ == "__main__":
    main()

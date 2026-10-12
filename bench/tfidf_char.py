"""Character n-gram TF-IDF (3–5-grams inside word boundaries) + linear SVM, and its ensemble with the
embedding LogReg. Character n-grams cope with compounds and inflection (German) better than word n-grams.

Methods: "tfidf_char" and "ensemble_char"; same protocol as extras.py (fit on train, choose C, w and T on
validation, score test once). Features are hashed (2^20 buckets) to bound memory; texts are cut to the
first MAX_CHARS characters for this model.

usage: python tfidf_char.py [dataset ...]
"""
import json
import sys
import time

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import store
from data import DATA_DIR, SPECS
from embed import EMB_DIR, MAX_TOKENS
from evaluate import RESULTS, metrics
from extras import TEMPERATURES, WEIGHTS

C_GRID = (0.1, 0.5, 1.0, 2.0)
MAX_CHARS = 5000


def features(texts_by_split):
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False,
                           norm=None, lowercase=True, dtype=np.float32)
    raw = {s: hv.transform([t[:MAX_CHARS] for t in texts]) for s, texts in texts_by_split.items()}
    tf = TfidfTransformer(sublinear_tf=True).fit(raw["train"])
    return {s: tf.transform(m) for s, m in raw.items()}


def run(name, chosen):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    t0 = time.time()
    X = features({s: [r["text"] for r in d[s]] for s in ("train", "val", "test")})
    table, best = [], None
    for C in C_GRID:
        t1 = time.time()
        svm = LinearSVC(C=C).fit(X["train"], y)
        acc = accuracy_score(yv, svm.predict(X["val"]))
        table.append({"params": {"C": C}, "val_acc": acc})
        print(f"  tfidf_char C={C}: val {acc:.4f} ({time.time() - t1:.0f}s)", flush=True)
        if best is None or acc > best[1]:
            best = (svm, acc, {"C": C})
    svm = best[0]
    pred_char = svm.predict(X["test"])
    res_char = {**metrics(yt, pred_char, n), "params": best[2], "val_acc": best[1], "selection": table,
                "features": "char_wb 3-5-grams, hashed to 2^20, sublinear tf-idf", "fit_seconds": round(time.time() - t0)}
    print(f"  tfidf_char test {res_char['accuracy']:.4f} mcc {res_char['mcc']:.4f}", flush=True)

    emb = np.load(EMB_DIR / f"{name}.npz")
    lr = LogisticRegression(C=chosen["logreg"]["C"], max_iter=1000).fit(emb[f"train_{MAX_TOKENS}"], y)
    pv, pt = lr.predict_proba(emb[f"val_{MAX_TOKENS}"]), lr.predict_proba(emb[f"test_{MAX_TOKENS}"])
    dv, dt = svm.decision_function(X["val"]), svm.decision_function(X["test"])
    if n == 2:  # binary decision_function is 1-d
        dv, dt = np.stack([-dv, dv], 1), np.stack([-dt, dt], 1)
    grid = [{"params": {"w": w, "T": T}, "val_acc": accuracy_score(yv, (w * pv + (1 - w) * softmax(T * dv, axis=1)).argmax(1))}
            for w in WEIGHTS for T in TEMPERATURES]
    top = max(grid, key=lambda g: g["val_acc"])
    w, T = top["params"]["w"], top["params"]["T"]
    proba = w * pt + (1 - w) * softmax(T * dt, axis=1)
    pred_ens = proba.argmax(1)
    per_w = [max((g for g in grid if g["params"]["w"] == wv), key=lambda g: g["val_acc"]) for wv in WEIGHTS]
    res_ens = {**metrics(yt, pred_ens, n), "params": top["params"], "val_acc": top["val_acc"], "selection": per_w}
    print(f"  ensemble_char w={w} T={T}: val {top['val_acc']:.4f} test {res_ens['accuracy']:.4f}", flush=True)
    top3 = np.argsort(-proba, axis=1)[:, :3]

    def merge(res, load):
        ds = res["datasets"][name]
        ds["methods"]["tfidf_char"] = res_char
        ds["methods"]["ensemble_char"] = res_ens
        p = load(name)
        p["pred"]["tfidf_char"] = pred_char.tolist()
        p["pred"]["ensemble_char"] = pred_ens.tolist()
        p["top3"]["ensemble_char"] = top3.tolist()
        p["top3p"]["ensemble_char"] = np.take_along_axis(proba, top3, 1).round(4).tolist()
    store.update(merge)


def main():
    results = json.loads(RESULTS.read_text())
    names = sys.argv[1:] or [n for n in SPECS if n in results["datasets"]]
    for name in names:
        print(name, flush=True)
        run(name, {"logreg": results["datasets"][name]["methods"]["logreg"]["params"]})


if __name__ == "__main__":
    main()

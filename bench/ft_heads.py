"""Classifiers on the fine-tuned encoder's embeddings (methods "ftemb_*").

The frozen-embedding sweep picks the classifier on top of the untouched encoder; this runs the same sweep on top of
each saved LoRA fine-tuned encoder instead: its pooled, L2-normalised embeddings for train / val / test, then kNN,
logistic regression, RBF SVM and a tuned MLP with the usual grids, chosen on validation, test scored once. It answers
whether, after fine-tuning, a different classifier than the jointly trained linear head is worth using.

Caveat: the encoder saw the training texts (and its checkpoint was chosen on validation), so the training embeddings
are tighter than unseen ones; kNN and the SVM profit most from that. Test texts were never seen.

Embeddings are cached in embeddings/<dataset>-finetuned.npz.

usage: python ft_heads.py [dataset ...]     default: every dataset with a saved fine-tuned model
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import mlp_search
import store
from data import DATA_DIR
from embed import EMB_DIR, wait_for_vram
from evaluate import GRIDS, RESULTS, fit_logreg, knn, metrics, select
from extras import SVM_MAX_TRAIN
from finetune import MODELS_DIR, batches, encode_texts, load_finetuned

SVM_C = (1.0, 10.0, 100.0)
MLP_TRIALS = 20


@torch.no_grad()
def finetuned_embeddings(model, tok, info, texts, batch_size=32):
    enc = encode_texts(tok, [info["prompt"] + t for t in texts], info["max_tokens"], info.get("truncation", "head"))
    lengths = [len(e) for e in enc]
    device = next(model.parameters()).device
    out = np.zeros((len(texts), model.head.in_features), dtype=np.float32)
    for ids in batches(lengths, batch_size, None, shuffle=False):
        width = max(lengths[i] for i in ids)
        input_ids = torch.full((len(ids), width), tok.pad_token_id)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, i in enumerate(ids):
            input_ids[r, :lengths[i]] = torch.tensor(enc[i])
            mask[r, :lengths[i]] = 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[ids] = model.embed(input_ids.to(device), mask.to(device)).float().cpu().numpy()
    return out


def embeddings(name, texts):
    path = EMB_DIR / f"{name}-finetuned.npz"
    if path.exists():
        e = np.load(path)
        return {s: e[s] for s in ("train", "val", "test")}
    wait_for_vram()
    _, model, tok, info = load_finetuned(MODELS_DIR / name)
    t0 = time.time()
    e = {s: finetuned_embeddings(model, tok, info, texts[s]) for s in ("train", "val", "test")}
    print(f"  embedded with the fine-tuned encoder in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    np.savez(path, **e)
    return e


def run(name):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    e = embeddings(name, texts)
    x, xv, xt = e["train"], e["val"], e["test"]
    out = {}

    def record(key, params, table, pred, extra=None):
        out[key] = {**metrics(yt, pred, n), "params": params, "val_acc": max(r["val_acc"] for r in table), "selection": table,
                    "embeddings": "fine-tuned encoder", **(extra or {})}
        print(f"  {key:14s} test {out[key]['accuracy']:.4f} {params}", flush=True)

    p, _, table = select("knn", GRIDS["knn"], lambda p: (p, accuracy_score(yv, knn(p, x, y, xv, n)), {}))
    record("ftemb_knn", p, table, knn(p, x, y, xt, n))

    def score_logreg(p):
        clf = fit_logreg(p, x, y)
        return clf, accuracy_score(yv, clf.predict(xv)), {}
    p, clf, table = select("logreg", GRIDS["logreg"], score_logreg)
    record("ftemb_logreg", p, table, clf.predict(xt))

    sx, sy, sub = x, y, None
    if len(y) > SVM_MAX_TRAIN:
        from sklearn.model_selection import train_test_split
        sx, _, sy, _ = train_test_split(x, y, train_size=SVM_MAX_TRAIN, stratify=y, random_state=0)
        sub = SVM_MAX_TRAIN

    def score_svm(p):
        clf = SVC(C=p["C"], kernel="rbf", gamma="scale").fit(sx, sy)
        return clf, accuracy_score(yv, clf.predict(xv)), {}
    p, clf, table = select("svm_rbf", [{"C": c} for c in SVM_C], score_svm)
    record("ftemb_svm", p, table, clf.predict(xt), {"train_subsample": sub} if sub else None)

    best, model, trials = mlp_search.search(x, y, xv, yv, n, n_trials=MLP_TRIALS, log=lambda *a: None)
    table = [{"params": t["params"], "val_acc": t["val_acc"]} for t in trials]
    record("ftemb_mlp", best, table, mlp_search.proba(model, xt).argmax(1), {"trials": len(trials)})

    def merge(res, load):
        res["datasets"][name]["methods"].update(out)
    store.update(merge)


def main():
    results = json.loads(RESULTS.read_text())
    names = sys.argv[1:] or [n for n in results["datasets"] if (MODELS_DIR / n / "classifier.json").exists()]
    for name in names:
        print(name, flush=True)
        run(name)


if __name__ == "__main__":
    main()

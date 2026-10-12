"""Ensemble of each saved fine-tuned encoder with the TF-IDF linear SVM (method "ft_ensemble").

Same protocol as the embedding ensemble in extras.py: p = w·p_finetuned + (1−w)·softmax(T·svm_scores),
w and T chosen on validation, test scored once. Needs the saved models in models/<dataset>/.

usage: python ft_ensemble.py [dataset ...]     default: every dataset with a saved fine-tuned model
"""
import json
import sys

import numpy as np
import torch
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import gpu
import store
from data import DATA_DIR, rel
from embed import wait_for_vram
from evaluate import RESULTS, metrics
from extras import TEMPERATURES, WEIGHTS
from finetune import MODELS_DIR, batches, encode_texts, load_finetuned


@torch.no_grad()
def finetuned_proba(model, tok, info, texts, batch_size=32):
    enc = encode_texts(tok, [info["prompt"] + t for t in texts], info["max_tokens"], info.get("truncation", "head"))
    lengths = [len(e) for e in enc]
    out = np.zeros((len(texts), len(info["labels"])), dtype=np.float32)
    for ids in batches(lengths, batch_size, None, shuffle=False):
        width = max(lengths[i] for i in ids)
        input_ids = torch.full((len(ids), width), tok.pad_token_id)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, i in enumerate(ids):
            input_ids[r, :lengths[i]] = torch.tensor(enc[i])
            mask[r, :lengths[i]] = 1
        with gpu.autocast():
            logits = model(input_ids.to(gpu.device()), mask.to(gpu.device()))
        out[ids] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def run(name, results):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))

    wait_for_vram()
    _, model, tok, info = load_finetuned(MODELS_DIR / name)
    assert info["labels"] == labels, "saved model was trained with a different label order"
    pv, pt = finetuned_proba(model, tok, info, texts["val"]), finetuned_proba(model, tok, info, texts["test"])
    del model
    gpu.empty_cache()
    print(f"  fine-tuned alone: val {accuracy_score(yv, pv.argmax(1)):.4f} test {accuracy_score(yt, pt.argmax(1)):.4f}", flush=True)

    C = results["datasets"][name]["methods"]["tfidf_svm"]["params"]["C"]
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    svm = LinearSVC(C=C).fit(vec.fit_transform(texts["train"]), y)
    dv, dt = svm.decision_function(vec.transform(texts["val"])), svm.decision_function(vec.transform(texts["test"]))
    grid = [{"params": {"w": w, "T": T}, "val_acc": accuracy_score(yv, (w * pv + (1 - w) * softmax(T * dv, axis=1)).argmax(1))}
            for w in WEIGHTS for T in TEMPERATURES]
    top = max(grid, key=lambda g: g["val_acc"])
    w, T = top["params"]["w"], top["params"]["T"]
    proba = w * pt + (1 - w) * softmax(T * dt, axis=1)
    pred = proba.argmax(1)
    per_w = [max((g for g in grid if g["params"]["w"] == wv), key=lambda g: g["val_acc"]) for wv in WEIGHTS]
    result = {**metrics(yt, pred, n), "params": top["params"], "val_acc": top["val_acc"], "selection": per_w,
              "parts": {"finetune": rel(MODELS_DIR / name), "tfidf_C": C}}
    print(f"  ft_ensemble w={w} T={T}: val {top['val_acc']:.4f} test {result['accuracy']:.4f}", flush=True)
    top3 = np.argsort(-proba, axis=1)[:, :3]

    def merge(res, load):
        res["datasets"][name]["methods"]["ft_ensemble"] = result
        p = load(name)
        p["pred"]["ft_ensemble"] = pred.tolist()
        p["top3"]["ft_ensemble"] = top3.tolist()
        p["top3p"]["ft_ensemble"] = np.take_along_axis(proba, top3, 1).round(4).tolist()
    store.update(merge)


def main():
    results = json.loads(RESULTS.read_text())
    names = sys.argv[1:] or [n for n in results["datasets"] if (MODELS_DIR / n / "classifier.json").exists()]
    for name in names:
        print(name, flush=True)
        run(name, results)


if __name__ == "__main__":
    main()

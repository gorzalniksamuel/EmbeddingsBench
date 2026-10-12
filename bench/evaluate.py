"""Compare classifiers on frozen Jina v5 classification embeddings, plus a TF-IDF baseline.

Protocol: every model is fit on train; hyperparameters and MLP early stopping use validation only; test is
scored once with the selected model. Main results use the 2,048-token embeddings; a 512-token ablation
re-scores the embedding classifiers with the same hyperparameters.

Writes results/results.json and results/predictions/<dataset>.json (per test item, for the explorer).
"""
import gzip
import json
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)
from sklearn.svm import LinearSVC

from data import DATA_DIR, MAX_CHARS, SEED, SPECS
from embed import ABLATION_TOKENS, EMB_DIR, LABEL_VARIANTS, MAX_TOKENS, MODEL_ID, REVISION

RESULTS = DATA_DIR.parent / "results" / "results.json"
PRED_DIR = RESULTS.parent / "predictions"
METHODS = ("zero_shot", "knn", "logreg", "mlp", "tfidf_svm")
GRIDS = {
    "zero_shot": [{"labels": k} for k in LABEL_VARIANTS],
    "knn": [{"k": k} for k in (1, 5, 10, 25, 50)],
    "logreg": [{"C": c} for c in (0.1, 1.0, 10.0, 100.0)],
    "mlp": [{"hidden": h, "dropout": p} for h in ((512,), (1024, 512)) for p in (0.1, 0.3)],
    "tfidf_svm": [{"C": c} for c in (0.1, 0.5, 1.0, 2.0)],
}
SHOTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
N_NEIGHBOURS = 5
SNIPPET = 400
import gpu  # noqa: E402

DEVICE = gpu.device()


# ---------------------------------------------------------------- classifiers
def zero_shot(params, emb, x):
    return (x @ emb[f"labels_{params['labels']}"].T).argmax(1), None


def knn(params, x, y, xq, n_classes):
    """Similarity-weighted vote of the k nearest training texts (cosine)."""
    k = params["k"]
    preds = []
    for start in range(0, len(xq), 2048):
        sims = xq[start:start + 2048] @ x.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < len(x) else np.argsort(-sims, axis=1)
        rows = np.arange(len(top))[:, None]
        votes = np.zeros((len(top), n_classes))
        np.add.at(votes, (np.repeat(rows, top.shape[1], 1), y[top]), sims[rows, top])
        preds.append(votes.argmax(1))
    return np.concatenate(preds)


def neighbours(x, xq, n=N_NEIGHBOURS):
    idx, sim = [], []
    for start in range(0, len(xq), 2048):
        sims = xq[start:start + 2048] @ x.T
        top = np.argpartition(-sims, n, axis=1)[:, :n]
        rows = np.arange(len(top))[:, None]
        top = top[rows, np.argsort(-sims[rows, top], axis=1)]
        idx.append(top)
        sim.append(sims[rows, top])
    return np.concatenate(idx), np.concatenate(sim)


def fit_logreg(params, x, y):
    return LogisticRegression(C=params["C"], max_iter=1000).fit(x, y)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden, n_classes, dropout):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp(params, x, y, xv, yv, n_classes, seed=SEED, max_epochs=300, patience=10):
    """AdamW + cross-entropy; early stopping on validation loss, best epoch restored."""
    torch.manual_seed(seed)
    model = MLP(x.shape[1], params["hidden"], n_classes, params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    xt, yt = torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)
    xvt, yvt = torch.tensor(xv, device=DEVICE), torch.tensor(yv, device=DEVICE)
    batch = min(256, max(8, len(x) // 4))
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(batch):
            idx = idx.to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.cross_entropy(model(xvt), yvt).item()
        if val_loss < best - 1e-4:
            best, bad, epochs = val_loss, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    model.epochs = epochs
    return model


def mlp_proba(model, x):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(x, device=DEVICE)), 1).cpu().numpy()


# ---------------------------------------------------------------- helpers
def metrics(y_true, y_pred, n):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(n), zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {"precision": p.round(4).tolist(), "recall": r.round(4).tolist(),
                      "f1": f.round(4).tolist(), "support": s.tolist()},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(n)).tolist(),
    }


def select(name, candidates, score):
    """Score every candidate on validation; return (best params, model, [{params, val_acc}])."""
    table, best = [], None
    for params in candidates:
        t0 = time.time()
        model, val_acc, extra = score(params)
        table.append({"params": params, "val_acc": val_acc, **extra})
        print(f"    {name:9s} {json.dumps(params):40s} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or val_acc > best[2]:
            best = (params, model, val_acc)
    return best[0], best[1], table


def run_embedding_methods(emb, tok, y, yv, n, fixed=None):
    """Fit/select zero-shot, kNN, LogReg and MLP on one embedding window. Returns {method: (params, model, table)}."""
    x, xv = emb[f"train_{tok}"], emb[f"val_{tok}"]
    grid = (lambda m: [fixed[m]]) if fixed else (lambda m: GRIDS[m])
    out = {}
    out["zero_shot"] = select("zero_shot", grid("zero_shot"),
                              lambda p: (p, accuracy_score(yv, zero_shot(p, emb, xv)[0]), {}))
    out["knn"] = select("knn", grid("knn"), lambda p: (p, accuracy_score(yv, knn(p, x, y, xv, n)), {}))

    def score_logreg(p):
        clf = fit_logreg(p, x, y)
        return clf, accuracy_score(yv, clf.predict(xv)), {}
    out["logreg"] = select("logreg", grid("logreg"), score_logreg)

    def score_mlp(p):
        model = fit_mlp(p, x, y, xv, yv, n)
        return model, accuracy_score(yv, mlp_proba(model, xv).argmax(1)), {"epochs": model.epochs}
    out["mlp"] = select("mlp", grid("mlp"), score_mlp)
    return out


def predict(method, params, model, emb, tok, y, n):
    x, xt = emb[f"train_{tok}"], emb[f"test_{tok}"]
    if method == "zero_shot":
        return zero_shot(params, emb, xt)[0], None
    if method == "knn":
        return knn(params, x, y, xt, n), None
    proba = model.predict_proba(xt) if method == "logreg" else mlp_proba(model, xt)
    return proba.argmax(1), proba


def learning_curve(emb, y, yv, yt, sel, n):
    """LogReg and MLP (selected hyperparameters) trained on k random examples per class."""
    x, xv, xt = emb[f"train_{MAX_TOKENS}"], emb[f"val_{MAX_TOKENS}"], emb[f"test_{MAX_TOKENS}"]
    rng = np.random.default_rng(SEED)
    by_class = [np.flatnonzero(y == c) for c in range(n)]
    smallest = min(len(i) for i in by_class)
    curve = []
    for k in [s for s in SHOTS if s < smallest] + ["all"]:
        draws = 3 if k != "all" and k < 32 else 1
        accs = {"logreg": [], "mlp": []}
        for d in range(draws):
            idx = np.arange(len(y)) if k == "all" else np.concatenate([rng.choice(i, k, replace=False) for i in by_class])
            accs["logreg"].append(accuracy_score(yt, fit_logreg(sel["logreg"], x[idx], y[idx]).predict(xt)))
            m = fit_mlp(sel["mlp"], x[idx], y[idx], xv, yv, n, seed=SEED + d)
            accs["mlp"].append(accuracy_score(yt, mlp_proba(m, xt).argmax(1)))
        point = {"shots": k if k != "all" else int(np.bincount(y).mean()), "all": k == "all",
                 "n_train": int(len(y) if k == "all" else k * n),
                 **{m: float(np.mean(a)) for m, a in accs.items()}}
        curve.append(point)
        print(f"    curve k={k}: logreg={point['logreg']:.4f} mlp={point['mlp']:.4f}", flush=True)
    return curve


# ---------------------------------------------------------------- main
def evaluate(name):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    emb = np.load(EMB_DIR / f"{name}.npz")
    meta, labels = d["meta"], d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}

    print(f"  [{MAX_TOKENS} tokens]", flush=True)
    chosen = run_embedding_methods(emb, MAX_TOKENS, y, yv, n)

    def score_svm(p):
        clf = LinearSVC(C=p["C"]).fit(xtr, y)
        return clf, accuracy_score(yv, clf.predict(xva)), {}
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    xtr, xva = vec.fit_transform(texts["train"]), vec.transform(texts["val"])
    chosen["tfidf_svm"] = select("tfidf_svm", GRIDS["tfidf_svm"], score_svm)

    preds, proba, results = {}, {}, {}
    for m in METHODS:
        params, model, table = chosen[m]
        if m == "tfidf_svm":
            preds[m] = model.predict(vec.transform(texts["test"]))
        else:
            preds[m], proba[m] = predict(m, params, model, emb, MAX_TOKENS, y, n)
        results[m] = {**metrics(yt, preds[m], n), "params": params,
                      "val_acc": max(r["val_acc"] for r in table), "selection": table}
        print(f"  {m:10s} test acc={results[m]['accuracy']:.4f} macroF1={results[m]['macro_f1']:.4f} mcc={results[m]['mcc']:.4f} {params}", flush=True)

    # 512-token ablation (same hyperparameters); identical inputs when no text exceeds 512 tokens
    tok_all = np.concatenate([emb[f"tokens_{s}"] for s in ("train", "val", "test")])
    ablation = {}
    if (tok_all > ABLATION_TOKENS).any():
        print(f"  [{ABLATION_TOKENS} tokens ablation]", flush=True)
        short = run_embedding_methods(emb, ABLATION_TOKENS, y, yv, n, fixed={m: chosen[m][0] for m in chosen})
        for m, (params, model, _) in short.items():
            p, _ = predict(m, params, model, emb, ABLATION_TOKENS, y, n)
            ablation[m] = {"accuracy": accuracy_score(yt, p), "macro_f1": f1_score(yt, p, average="macro"),
                           "mcc": float(matthews_corrcoef(yt, p))}
    else:
        ablation = {m: {k: results[m][k] for k in ("accuracy", "macro_f1", "mcc")} for m in METHODS if m != "tfidf_svm"}

    print("  learning curve", flush=True)
    curve = learning_curve(emb, y, yv, yt, {m: chosen[m][0] for m in ("logreg", "mlp")}, n)

    x = emb[f"train_{MAX_TOKENS}"]
    nn_idx, nn_sim = neighbours(x, emb[f"test_{MAX_TOKENS}"])
    tokens = emb["tokens_test"]
    payload = {
        "text": texts["test"], "y": yt.tolist(), "tokens": tokens.tolist(),
        "pred": {m: p.tolist() for m, p in preds.items()},
        "top3": {}, "top3p": {},
        "nn": nn_idx.tolist(), "nn_sim": nn_sim.round(4).tolist(),
        "train_y": y.tolist(), "train_snippet": [t[:SNIPPET] for t in texts["train"]],
    }
    for m in ("logreg", "mlp"):
        top = np.argsort(-proba[m], axis=1)[:, :3]
        payload["top3"][m] = top.tolist()
        payload["top3p"][m] = np.take_along_axis(proba[m], top, 1).round(4).tolist()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    (PRED_DIR / f"{name}.json").write_bytes(raw)
    (PRED_DIR / f"{name}.json.gz").write_bytes(gzip.compress(raw, 6))

    counts = lambda a: np.bincount(a, minlength=n).tolist()
    tok = {s: emb[f"tokens_{s}"] for s in ("train", "val", "test")}
    return {
        "meta": {**meta, "counts": {"train": counts(y), "val": counts(yv), "test": counts(yt)},
                 "lengths": length_stats(texts, tok),
                 "tokens": {s: {"median": int(np.median(t)), "p90": int(np.percentile(t, 90)),
                                "over_512": float(np.mean(t > ABLATION_TOKENS)),
                                "over_max": float(np.mean(t > MAX_TOKENS))} for s, t in tok.items()},
                 "ablation_identical": not (tok_all > ABLATION_TOKENS).any()},
        "methods": results,
        "ablation_512": ablation,
        "learning_curve": curve,
    }


def length_stats(texts, tok):
    """Average characters and tokens per split and overall."""
    out = {s: {"avg_chars": float(np.mean([len(t) for t in texts[s]])), "avg_tokens": float(np.mean(tok[s]))} for s in texts}
    allc = [len(t) for s in texts for t in texts[s]]
    allt = np.concatenate([np.asarray(tok[s]) for s in texts])
    out["all"] = {"avg_chars": float(np.mean(allc)), "avg_tokens": float(np.mean(allt))}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="evaluate only these and merge them into results.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.datasets and RESULTS.exists():
        import store
        for name in a.datasets:
            print(name, flush=True)
            entry = evaluate(name)
            def merge(res, load, name=name, entry=entry):
                res["datasets"][name] = entry
                res["complete"] = True
            store.update(merge)
        return
    out = {"model": MODEL_ID, "revision": REVISION, "max_tokens": MAX_TOKENS, "ablation_tokens": ABLATION_TOKENS,
           "max_chars": MAX_CHARS, "seed": SEED, "label_variants": LABEL_VARIANTS, "grids": GRIDS,
           "mlp": {"optimizer": "AdamW lr 1e-3, weight decay 1e-2", "activation": "GELU",
                   "early_stopping": "validation loss, patience 10, best epoch restored", "max_epochs": 300},
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "datasets": {}}
    for name in SPECS:
        print(name, flush=True)
        out["datasets"][name] = evaluate(name)
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(out))  # partial results after every dataset
    out["complete"] = True
    RESULTS.write_text(json.dumps(out))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()

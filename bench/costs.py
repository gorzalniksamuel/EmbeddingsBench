"""Parameter counts, measured training effort and fitted scaling laws for every dataset × classifier.

Measured times come from the run logs (one shared 4-core VM + RTX 3060, other services running, so treat
them as approximate). Fitted models:
  GPU embedding   t = a·Σtok + b·Σtok²/1000        tok = tokens per text, capped at the window
  GPU fine-tuning t = c·epochs·Σmin(tok, 512)      classes only change the head (1,024·C weights)
  CPU classifiers t = k·f(N, C) with f from the algorithm's complexity (free power-law fits over six datasets of
                  similar size were unstable), k calibrated as the median ratio; min/max ratio give the spread
Writes results.json["costs"].
"""
import json
import re

import numpy as np

import store
from data import DATA_DIR, SPECS
from embed import ABLATION_TOKENS, EMB_DIR, MAX_TOKENS
from evaluate import RESULTS

ROOT = DATA_DIR.parent
ENCODER_PARAMS = 606_000_000
DIM = 1024
LOGS = ["embed.log", "eval.log", "extras.log", "mlp_search.log", "add_datasets.log"]
TIME_RE = {
    "logreg": re.compile(r"^\s+logreg\s+(\{.*?\})\s+val_acc=\S+ \((\d+)s\)"),
    "mlp": re.compile(r"^\s+mlp\s+(\{.*?\})\s+val_acc=\S+ \((\d+)s\)"),
    "tfidf_svm": re.compile(r"^\s+tfidf_svm\s+(\{.*?\})\s+val_acc=\S+ \((\d+)s\)"),
    "knn": re.compile(r"^\s+knn\s+(\{.*?\})\s+val_acc=\S+ \((\d+)s\)"),
    "svm_rbf": re.compile(r"^\s+svm_rbf C=(\S+): val \S+ \((\d+)s\)"),
}
EMBED_DONE = re.compile(r"^(\w+) done (\d+)s")


def parse_logs():
    """{dataset: {"embedding_s": s, "fit_s": {method: {params-json: seconds}}}} from all logs."""
    out = {n: {"fit_s": {}} for n in SPECS}
    for log in LOGS:
        path = ROOT / log
        if not path.exists():
            continue
        current = None
        for line in path.read_text(errors="replace").splitlines():
            if line.strip() in SPECS and not line.startswith(" "):
                current = line.strip()
                continue
            m = EMBED_DONE.match(line)
            if m and m[1] in SPECS:
                out[m[1]]["embedding_s"] = int(m[2])
                continue
            if current is None or "[512 tokens ablation]" in line:
                current = None if "[512 tokens ablation]" in line else current
                continue
            for method, rx in TIME_RE.items():
                m = rx.match(line)
                if m:
                    key = json.dumps({"C": float(m[1])}) if method == "svm_rbf" else m[1]
                    out[current]["fit_s"].setdefault(method, {})[key] = int(m[2])
    return out


def token_sums(name):
    emb = np.load(EMB_DIR / f"{name}.npz")
    t = np.concatenate([emb[f"tokens_{s}"] for s in ("train", "val", "test")])
    main = np.minimum(t, MAX_TOKENS)
    long_ = t[t > ABLATION_TOKENS]
    # the run embeds everything at 2,048 and re-embeds long texts at 512 for the ablation
    lin = float(main.sum() + ABLATION_TOKENS * len(long_))
    quad = float((main.astype(np.float64) ** 2).sum() + len(long_) * ABLATION_TOKENS ** 2) / 1000
    train_ft = float(np.minimum(emb["tokens_train"], 512).sum())
    return {"texts": int(len(t)), "tokens": lin, "tokens_sq_k": quad, "train_tokens_512": train_ft,
            "train_texts": int(len(emb["tokens_train"]))}


def params(method, entry, n_classes, n_train):
    """Trainable parameters and stored numbers per classifier (benchmark settings)."""
    p = entry["methods"].get(method, {}).get("params") or {}
    if method == "zero_shot":
        return {"trainable": 0, "stored": n_classes * DIM, "encoder": ENCODER_PARAMS}
    if method == "knn":
        return {"trainable": 0, "stored": n_train * DIM, "encoder": ENCODER_PARAMS}
    if method == "logreg":
        return {"trainable": n_classes * (DIM + 1), "stored": n_classes * (DIM + 1), "encoder": ENCODER_PARAMS}
    if method == "mlp":
        dims = [DIM, *p.get("hidden", []), n_classes]
        t = sum(a * b + b for a, b in zip(dims, dims[1:]))
        return {"trainable": t, "stored": t, "encoder": ENCODER_PARAMS}
    if method == "mlp_tuned":
        t = entry["methods"]["mlp_tuned"].get("n_params")
        return {"trainable": t, "stored": t, "encoder": ENCODER_PARAMS}
    if method == "finetune":
        c = entry["methods"]["finetune"]["config"]
        return {"trainable": c["trainable_params"], "stored": c["trainable_params"], "encoder": c["total_params"]}
    if method.startswith("finetune_"):
        c = entry["methods"][method]["config"]
        return {"trainable": c["trainable_params"], "stored": c["trainable_params"], "encoder": c["total_params"]}
    if method == "gbert":
        c = entry["methods"]["gbert"]["config"]
        return {"trainable": c["params"], "stored": c["params"], "encoder": 0, "note": "deepset/gbert-large, all weights trained"}
    if method in ("tfidf_char", "ensemble_char"):
        t = 2 ** 20 * n_classes + n_classes + (n_classes * (DIM + 1) if method == "ensemble_char" else 0)
        return {"trainable": t, "stored": t, "encoder": ENCODER_PARAMS if method == "ensemble_char" else 0, "note": "hashed char 3–5-grams (2^20 buckets)"}
    if method == "ft_ensemble" and "finetune" in entry["methods"]:
        c = entry["methods"]["finetune"]["config"]
        return {"trainable": c["trainable_params"], "stored": c["trainable_params"], "encoder": c["total_params"],
                "note": "fine-tuned encoder + TF-IDF SVM (see TF-IDF row for its n-gram weights)"}
    return None  # tfidf_svm / svm_rbf / ensemble: filled from refits below


def refit_counts(name, entry):
    """TF-IDF vocabulary and RBF support vectors need a refit with the selected settings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    labels = d["meta"]["labels"]
    n = len(labels)
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    vec.fit([r["text"] for r in d["train"]])
    feats = len(vec.vocabulary_)
    out = {"tfidf_svm": {"trainable": feats * n + n, "stored": feats * n + n, "encoder": 0, "note": f"{feats:,} n-gram features"}}
    lr = n * (DIM + 1)
    out["ensemble"] = {"trainable": feats * n + n + lr, "stored": feats * n + n + lr, "encoder": ENCODER_PARAMS}
    if "svm_rbf" in entry["methods"]:
        index = {l: i for i, l in enumerate(labels)}
        y = np.array([index[r["label"]] for r in d["train"]])
        x = np.load(EMB_DIR / f"{name}.npz")[f"train_{MAX_TOKENS}"]
        svc = SVC(C=entry["methods"]["svm_rbf"]["params"]["C"], kernel="rbf", gamma="scale").fit(x, y)
        sv = int(svc.support_vectors_.shape[0])
        k = int(svc.dual_coef_.size + svc.intercept_.size)
        out["svm_rbf"] = {"trainable": k, "stored": sv * DIM + k, "encoder": ENCODER_PARAMS, "note": f"{sv:,} support vectors"}
    return out


LAWS = {  # method: (alpha on N, beta on C, why)
    "logreg": (1, 1, "L-BFGS softmax: every iteration touches each text × class weight"),
    "tfidf_svm": (1, 1, "linear SVM, one-vs-rest: one pass over the sparse texts per class"),
    "mlp": (1, 0, "per epoch ∝ texts × layer width; early stopping fixes the epoch count"),
    "svm_rbf": (2, 0, "kernel SVM: kernel values between pairs of training texts (one-vs-one keeps it ≈ N²)"),
}


def fit_law(method, rows):
    rows = [r for r in rows if r["seconds"] > 0]
    if method not in LAWS or len(rows) < 2:
        return None
    a, b, why = LAWS[method]
    ratios = np.array([r["seconds"] / (r["n"] ** a * r["c"] ** b) for r in rows])
    return {"alpha": a, "beta": b, "k": float(np.median(ratios)), "k_min": float(ratios.min()), "k_max": float(ratios.max()),
            "why": why, "points": len(rows)}


def main():
    results = json.loads(RESULTS.read_text())
    logs = parse_logs()
    per, emb_rows, ft_rows, cpu = {}, [], [], {m: [] for m in ("logreg", "mlp", "tfidf_svm", "svm_rbf", "knn")}
    refits = {}
    for name, entry in results["datasets"].items():
        meta = entry["meta"]
        n_classes, n_train = len(meta["labels"]), meta["n_train"]
        tok = token_sums(name)
        rec = {"tokens": tok, "embedding_s": logs[name].get("embedding_s"), "fit_s": {}, "params": {}}
        for method, table in logs[name]["fit_s"].items():
            chosen = entry["methods"].get(method, {}).get("params")
            key = json.dumps(chosen) if chosen is not None else None
            if key in table:
                rec["fit_s"][method] = table[key]
                cpu[method].append({"dataset": name, "n": n_train, "c": n_classes, "seconds": table[key]})
            rec["fit_s"][method + "_all_settings"] = sum(table.values())
        if "finetune" in entry["methods"]:
            c = entry["methods"]["finetune"]["config"]
            rec["fit_s"]["finetune"] = c["train_seconds"]
            ft_rows.append({"dataset": name, "units": c["epochs"] * tok["train_tokens_512"], "seconds": c["train_seconds"],
                            "n": n_train, "c": n_classes})
        if "mlp_tuned" in entry["methods"]:
            rec["fit_s"]["mlp_search"] = entry["methods"]["mlp_tuned"].get("search_seconds")
        if rec["embedding_s"]:
            emb_rows.append({"dataset": name, "lin": tok["tokens"], "quad": tok["tokens_sq_k"], "seconds": rec["embedding_s"]})
        print(name, "refit for parameter counts", flush=True)
        refits = refit_counts(name, entry)
        for method in entry["methods"]:
            rec["params"][method] = params(method, entry, n_classes, n_train) or refits.get(method)
        per[name] = rec

    fits = {}
    if len(emb_rows) >= 2:
        A = np.array([[r["lin"], r["quad"]] for r in emb_rows])
        b = np.array([r["seconds"] for r in emb_rows], dtype=float)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        if (coef < 0).any():  # fall back to purely linear in tokens
            coef = np.array([float((A[:, 0] @ b) / (A[:, 0] @ A[:, 0])), 0.0])
        fits["embedding"] = {"a": float(coef[0]), "b": float(coef[1]), "points": emb_rows,
                             "tokens_per_s_short": float(1 / coef[0]) if coef[0] > 0 else None}
    if ft_rows:
        u = np.array([r["units"] for r in ft_rows]); s = np.array([r["seconds"] for r in ft_rows], dtype=float)
        c = float((u @ s) / (u @ u))
        fits["finetune"] = {"c": c, "tokens_per_s": 1 / c, "points": ft_rows, "vram_gib": 3.6,
                            "flops_per_token": 8 * 606_000_000}
    fits["cpu"] = {m: {"fit": fit_law(m, rows), "points": rows} for m, rows in cpu.items()}
    out = {"per_dataset": per, "fits": fits, "encoder_params": ENCODER_PARAMS,
           "note": "Times measured on a shared 4-core VM with an RTX 3060 (12 GB) while other jobs ran; treat as approximate."}

    def merge(res, load):
        res["costs"] = out
    store.update(merge)
    print(json.dumps({k: v for k, v in fits.items() if k != "cpu"}, default=str)[:800])
    print({m: v["fit"] for m, v in fits["cpu"].items()})


if __name__ == "__main__":
    main()

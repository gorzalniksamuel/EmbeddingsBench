"""Model and data registry: versioned train/val/test splits and versioned models trained on them.

Layout under registry/:
  splits/<dataset>/<split_id>/split.json   config, stats, description (immutable once built)
  splits/<dataset>/<split_id>/data.json    the texts (benchmark split: points to data/<dataset>.json)
  emb/<dataset>/{keys.json,vectors.npy,tokens.npy}   embedding cache keyed by text hash (2,048 tokens)
  models/<name>/v<N>/model.json            config, status, metrics, description, tags, stage
  models/<name>/v<N>/...                   artifacts (joblib / torch / LoRA adapter) + predictions.json.gz

Jobs (build_split, train) are run by jobs.py, one at a time, queued by the web app or the CLI.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data import DATA_DIR, SEED, SPECS, build, load_custom_specs, official_val_split, rel, resolve

ROOT = DATA_DIR.parent
REG = ROOT / "registry"
SPLITS = REG / "splits"
MODELS = REG / "models"
EMB = REG / "emb"
ENCODER_PARAMS = 606_000_000  # measured parameter count of the loaded encoder (Qwen3-0.6B backbone)
EMBED_DIM = 1024
BUILTIN_METHODS = ("zero_shot", "knn", "logreg", "svm_rbf", "mlp", "tfidf_svm", "ensemble", "finetune")
from classifiers import Split, count_params, discover  # noqa: E402  (classifier plugins, see classifiers/__init__.py)

PLUGINS = discover()
METHODS = BUILTIN_METHODS + tuple(k for k in PLUGINS if k not in BUILTIN_METHODS)
METHOD_INFO = {k: {"label": m.LABEL, "description": m.DESCRIPTION, "input": m.INPUT,
                   "grid": m.GRID, "plugin": True} for k, m in PLUGINS.items()}
GPU_METHODS = ("finetune",)


def now():
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def text_key(text):
    return hashlib.sha1(text.encode()).hexdigest()


# ------------------------------------------------------------------ splits
def split_dir(dataset, split_id):
    return SPLITS / dataset / split_id


def load_split(dataset, split_id):
    meta = json.loads((split_dir(dataset, split_id) / "split.json").read_text())
    path = resolve(meta["data_path"])
    return meta, json.loads(path.read_text())


def split_stats(d, tokens):
    """Per-split counts, per-class counts and average lengths."""
    labels = d["meta"]["labels"]
    out = {}
    for s in ("train", "val", "test"):
        rows = d[s]
        counts = {l: 0 for l in labels}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        chars = np.array([len(r["text"]) for r in rows])
        tok = np.asarray(tokens[s])
        out[s] = {"n": len(rows), "per_class": [counts[l] for l in labels],
                  "avg_chars": float(chars.mean()), "avg_tokens": float(tok.mean()),
                  "median_tokens": float(np.median(tok)), "over_512": float((tok > 512).mean()), "over_2048": float((tok > 2048).mean())}
    return out


def ensure_benchmark_split(dataset):
    """Register the benchmark split (data/<dataset>.json) as split 'benchmark' and seed the embedding cache."""
    meta_path = split_dir(dataset, "benchmark") / "split.json"
    if meta_path.exists():
        return
    from embed import EMB_DIR, MAX_TOKENS
    if not (DATA_DIR / f"{dataset}.json").exists() or not (EMB_DIR / f"{dataset}.npz").exists():
        return  # defined but not built / embedded yet
    d = json.loads((DATA_DIR / f"{dataset}.json").read_text())
    emb = np.load(EMB_DIR / f"{dataset}.npz")
    texts = sum(([r["text"] for r in d[s]] for s in ("train", "val", "test")), [])
    vecs = np.concatenate([emb[f"{s}_{MAX_TOKENS}"] for s in ("train", "val", "test")])
    toks = np.concatenate([emb[f"tokens_{s}"] for s in ("train", "val", "test")])
    _cache_add(dataset, texts, vecs, toks)
    spec = SPECS[dataset]
    config = {"train_per_class": spec["train_per_class"], "test_per_class": spec["test_per_class"],
              "val": list(spec["val"]), "seed": SEED}
    write_json(meta_path, {
        "dataset": dataset, "id": "benchmark", "title": "Benchmark split", "created": "2026-10-11",
        "description": "The split used for the benchmark on the overview page. " + d["meta"]["split_note"],
        "config": config, "labels": d["meta"]["labels"], "label_text": d["meta"]["label_text"],
        "dupes_removed": d["meta"]["dupes_removed"], "data_path": rel(DATA_DIR / f"{dataset}.json"),
        "stats": split_stats(d, {s: emb[f"tokens_{s}"] for s in ("train", "val", "test")}), "status": "ready", "immutable": True,
    })


def list_splits(dataset):
    if dataset not in SPECS:
        return []
    ensure_benchmark_split(dataset)
    out = []
    if not (SPLITS / dataset).exists():
        return out
    for p in sorted((SPLITS / dataset).glob("*/split.json")):
        out.append(json.loads(p.read_text()))
    return out


def build_split(dataset, split_id, config, description, log=print):
    """Sample a new split with the given config, embed any texts not cached yet, store it."""
    ensure_benchmark_split(dataset)
    val = config["val"]
    overrides = {"train_per_class": config.get("train_per_class"), "test_per_class": config.get("test_per_class"),
                 "val": (val[0], val[1]), "seed": int(config.get("seed", SEED))}
    if overrides["val"][0] == "official" and not official_val_split(dataset):
        raise ValueError(f"{dataset} has no official validation split; use a hold-out")
    if overrides["val"][0] == "official":
        overrides["val"] = ("official", official_val_split(dataset))
    log(f"sampling {dataset} with {overrides}")
    meta, splits = build(dataset, overrides)
    d = {"meta": meta, **splits}
    out = split_dir(dataset, split_id)
    write_json(out / "data.json", d)
    tokens = {}
    for s in ("train", "val", "test"):
        log(f"{s}: {len(splits[s])} texts")
        _, tokens[s] = embeddings(dataset, [r["text"] for r in splits[s]], log)
    write_json(out / "split.json", {
        "dataset": dataset, "id": split_id, "title": config.get("title") or split_id, "created": now(),
        "description": description, "config": {**config, "val": list(overrides["val"])},
        "labels": meta["labels"], "label_text": meta["label_text"], "dupes_removed": meta["dupes_removed"],
        "data_path": rel(out / "data.json"), "stats": split_stats(d, tokens), "status": "ready", "immutable": True,
        "note": meta["split_note"],
    })
    log("split ready")


# ------------------------------------------------------------------ embedding cache
def _cache_paths(dataset):
    base = EMB / dataset
    return base / "keys.json", base / "vectors.npy", base / "tokens.npy"


def _cache_load(dataset):
    kp, vp, tp = _cache_paths(dataset)
    if not kp.exists():
        return {}, np.zeros((0, 1024), np.float32), np.zeros(0, np.int64)
    return json.loads(kp.read_text()), np.load(vp), np.load(tp)


def _cache_add(dataset, texts, vecs, toks):
    keys, V, T = _cache_load(dataset)
    new = [(text_key(t), i) for i, t in enumerate(texts) if text_key(t) not in keys]
    seen = set()
    new = [(k, i) for k, i in new if not (k in seen or seen.add(k))]
    if not new:
        return
    for j, (k, _) in enumerate(new):
        keys[k] = len(V) + j
    idx = [i for _, i in new]
    V = np.concatenate([V, np.asarray(vecs)[idx]])
    T = np.concatenate([T, np.asarray(toks)[idx]])
    kp, vp, tp = _cache_paths(dataset)
    kp.parent.mkdir(parents=True, exist_ok=True)
    np.save(vp, V)
    np.save(tp, T)
    write_json(kp, keys)


GPU_SECONDS = {"embedding": 0.0}  # GPU time spent embedding inside the current process


def embeddings(dataset, texts, log=print):
    """2,048-token embeddings and token counts for texts, embedding missing ones on the GPU."""
    keys, V, T = _cache_load(dataset)
    missing = [t for t in dict.fromkeys(texts) if text_key(t) not in keys]
    if missing:
        import gpu
        from embed import MAX_TOKENS, encode, load_encoder, token_lengths, wait_for_vram
        log(f"embedding {len(missing)} new texts on {gpu.describe().get('name', 'the CPU')}")
        wait_for_vram()
        t_emb = time.time()
        model = load_encoder()
        lengths = token_lengths(model, missing)
        vecs = encode(model, missing, lengths, MAX_TOKENS)
        del model
        gpu.empty_cache()
        _cache_add(dataset, missing, vecs, lengths)
        GPU_SECONDS["embedding"] += time.time() - t_emb
        log(f"embedded {len(missing)} texts ({int(np.minimum(lengths, MAX_TOKENS).sum()):,} tokens) in {time.time() - t_emb:.0f}s")
        keys, V, T = _cache_load(dataset)
    rows = [keys[text_key(t)] for t in texts]
    return V[rows], T[rows]


# ------------------------------------------------------------------ models
def list_models():
    out = []
    for p in sorted(MODELS.glob("*/v*/model.json")):
        out.append(json.loads(p.read_text()))
    return out


def model_dir(name, version):
    return MODELS / name / f"v{version}"


def update_model(name, version, **fields):
    path = model_dir(name, version) / "model.json"
    m = json.loads(path.read_text())
    m.update(fields)
    m["updated"] = now()
    write_json(path, m)
    return m


def train_model(name, version, log=print):
    """Train the model version described in its model.json (status queued) and store metrics + artifacts."""
    from sklearn.metrics import accuracy_score

    from evaluate import GRIDS, fit_logreg, fit_mlp, knn, metrics, mlp_proba
    m = json.loads((model_dir(name, version) / "model.json").read_text())
    out_dir = model_dir(name, version)
    dataset, split_id, method = m["dataset"], m["split"], m["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    smeta, d = load_split(dataset, split_id)
    labels = smeta["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}
    x, xv, xt = (embeddings(dataset, texts[s], log)[0] for s in ("train", "val", "test"))
    params = m.get("params") or {}
    t0 = time.time()
    selection, proba, pred_val = [], None, None

    def pick(candidates, fit_score):
        best = None
        for p in candidates:
            model, acc, extra = fit_score(p)
            selection.append({"params": p, "val_acc": acc, **extra})
            log(f"  {json.dumps(p)} → validation {acc:.4f}")
            if best is None or acc > best[2]:
                best = (p, model, acc)
        return best

    if method in PLUGINS:
        plugin = PLUGINS[method]
        splits = {k: Split(x=v, texts=texts[k], y=yy, n_classes=n) for k, v, yy in (("train", x, y), ("val", xv, yv), ("test", xt, yt))}
        cands = [{**plugin.GRID[0], **params}] if params else plugin.GRID
        def score(p):
            mdl = plugin.fit(p, splits["train"])
            return mdl, accuracy_score(yv, plugin.predict_proba(mdl, splits["val"]).argmax(1)), {}
        chosen, plugin_model, val_acc = pick(cands, score)
        proba = np.asarray(plugin.predict_proba(plugin_model, splits["test"]), dtype=np.float32)
        if hasattr(plugin, "save"):
            plugin.save(plugin_model, out_dir / "model")
        else:
            joblib.dump(plugin_model, out_dir / f"{method}.joblib")
    elif method == "zero_shot":
        from embed import LABEL_VARIANTS
        names = [smeta["label_text"][l] for l in labels]
        protos = {k: embeddings(dataset, [t.format(nm) for nm in names], log)[0] for k, t in LABEL_VARIANTS.items()}
        cands = [params] if params else [{"labels": k} for k in LABEL_VARIANTS]
        chosen, _, val_acc = pick(cands, lambda p: (None, accuracy_score(yv, (xv @ protos[p["labels"]].T).argmax(1)), {}))
        pred = (xt @ protos[chosen["labels"]].T).argmax(1)
        np.save(out_dir / "label_prototypes.npy", protos[chosen["labels"]])
    elif method == "knn":
        cands = [params] if params else GRIDS["knn"]
        chosen, _, val_acc = pick(cands, lambda p: (None, accuracy_score(yv, knn(p, x, y, xv, n)), {}))
        pred = knn(chosen, x, y, xt, n)
    elif method == "logreg":
        cands = [params] if params else GRIDS["logreg"]
        chosen, clf, val_acc = pick(cands, lambda p: (lambda c: (c, accuracy_score(yv, c.predict(xv)), {}))(fit_logreg(p, x, y)))
        proba = clf.predict_proba(xt)
        joblib.dump(clf, out_dir / "logreg.joblib")
    elif method == "svm_rbf":
        from sklearn.svm import SVC
        from extras import SVM_C
        cands = [params] if params else [{"C": c} for c in SVM_C]
        chosen, clf, val_acc = pick(cands, lambda p: (lambda c: (c, accuracy_score(yv, c.predict(xv)), {}))(SVC(C=p["C"], kernel="rbf", gamma="scale").fit(x, y)))
        pred = clf.predict(xt)
        joblib.dump(clf, out_dir / "svm_rbf.joblib")
    elif method == "mlp":
        import torch
        import mlp_search
        if params:  # manual: one configuration, early stopping on validation
            p = {**mlp_search.MANUAL_DEFAULTS, **params}
            mdl = mlp_search.fit(p, x, y, xv, yv, n)
            chosen, val_acc = p, accuracy_score(yv, mlp_search.proba(mdl, xv).argmax(1))
            selection.append({"params": p, "val_acc": val_acc, "epochs": mdl.epochs,
                              "train_acc": accuracy_score(y, mlp_search.proba(mdl, x).argmax(1)), "n_params": mlp_search.n_params(mdl)})
        else:  # automatic bias–variance search on validation
            chosen, mdl, trials = mlp_search.search(x, y, xv, yv, n, n_trials=int(params.get("trials", 30)), log=log)
            val_acc = max(t["val_acc"] for t in trials)
            selection.extend({"params": t["params"], "val_acc": t["val_acc"], "train_acc": t["train_acc"],
                              "n_params": t["n_params"], "epochs": t["epochs"]} for t in trials)
            m["mlp_search"] = trials
        proba = mlp_search.proba(mdl, xt)
        torch.save({"state_dict": mdl.state_dict(), "params": chosen, "dim": x.shape[1], "n_classes": n}, out_dir / "mlp.pt")
    elif method in ("tfidf_svm", "ensemble"):
        from scipy.special import softmax
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
        xtr = vec.fit_transform(texts["train"])
        xva, xte = vec.transform(texts["val"]), vec.transform(texts["test"])
        if method == "tfidf_svm":
            cands = [params] if params else GRIDS["tfidf_svm"]
            chosen, clf, val_acc = pick(cands, lambda p: (lambda c: (c, accuracy_score(yv, c.predict(xva)), {}))(LinearSVC(C=p["C"]).fit(xtr, y)))
            pred = clf.predict(xte)
            joblib.dump({"vectorizer": vec, "svm": clf}, out_dir / "tfidf_svm.joblib")
        else:
            from extras import TEMPERATURES, WEIGHTS
            emb_c = params.get("logreg_C") or pick_quick(GRIDS["logreg"], lambda p: accuracy_score(yv, fit_logreg(p, x, y).predict(xv)))["C"]
            svm_c = params.get("tfidf_C") or pick_quick(GRIDS["tfidf_svm"], lambda p: accuracy_score(yv, LinearSVC(C=p["C"]).fit(xtr, y).predict(xva)))["C"]
            lr = fit_logreg({"C": emb_c}, x, y)
            svm = LinearSVC(C=svm_c).fit(xtr, y)
            pv, dv = lr.predict_proba(xv), svm.decision_function(xva)
            cands = [{"w": params["w"], "T": params["T"]}] if "w" in params else [{"w": w, "T": T} for w in WEIGHTS for T in TEMPERATURES]
            chosen, _, val_acc = pick(cands, lambda p: (None, accuracy_score(yv, (p["w"] * pv + (1 - p["w"]) * softmax(p["T"] * dv, axis=1)).argmax(1)), {}))
            chosen = {**chosen, "logreg_C": emb_c, "tfidf_C": svm_c}
            proba = chosen["w"] * lr.predict_proba(xt) + (1 - chosen["w"]) * softmax(chosen["T"] * svm.decision_function(xte), axis=1)
            joblib.dump({"vectorizer": vec, "svm": svm, "logreg": lr, "w": chosen["w"], "T": chosen["T"]}, out_dir / "ensemble.joblib")
    elif method == "finetune":
        import finetune
        C = params.get("head_C") or pick_quick(GRIDS["logreg"], lambda p: accuracy_score(yv, fit_logreg(p, x, y).predict(xv)))["C"]
        opts = {k: params[k] for k in finetune.DEFAULTS if k in params}
        d_named = {**d, "meta": {**d["meta"], "name": dataset}}
        res = finetune.train(d_named, x, C, opts, out_dir / "finetune", log=log)
        proba, chosen, val_acc = res["proba"], {**finetune.DEFAULTS, **opts, "head_C": C}, res["result"]["val_acc"]
        selection = res["result"]["selection"]
        m["history"] = res["result"]["history"]
        m["finetune_config"] = res["result"]["config"]
    if proba is not None:
        pred = proba.argmax(1)
    train_seconds = round(time.time() - t0)
    obj = {"logreg": locals().get("clf"), "svm_rbf": locals().get("clf"), "tfidf_svm": locals().get("clf"), "mlp": locals().get("mdl"),
           **({method: locals().get("plugin_model")} if method in PLUGINS else {})}.get(method)
    if method == "ensemble":
        obj = (lr, svm)
    tok_train = embeddings(dataset, texts["train"], log)[1]
    params_info = param_count(method, obj, n, len(y), m.get("finetune_config"))
    if method in PLUGINS and PLUGINS[method].INPUT == "text":
        params_info["encoder"] = 0
    effort = {"device": "GPU" if method in GPU_METHODS else "CPU", "train_seconds": train_seconds,
              "embedding_gpu_seconds": round(GPU_SECONDS["embedding"]), "train_texts": int(len(y)),
              "train_tokens": int(np.minimum(tok_train, 2048).sum())}
    if method == "finetune":
        fc = m["finetune_config"]
        tt = int(np.minimum(tok_train, fc["max_tokens"]).sum())
        effort["train_tokens"] = tt
        # forward 2·P·T + backward ~4·P·T + checkpoint recompute 2·P·T per training token, per epoch
        effort["est_train_flops"] = float(8 * fc["total_params"] * tt * fc["epochs"])
    test = metrics(yt, pred, n)
    preds = {"y": yt.tolist(), "pred": pred.tolist()}
    if proba is not None:
        top = np.argsort(-proba, axis=1)[:, :3]
        preds.update(top3=top.tolist(), top3p=np.take_along_axis(proba, top, 1).round(4).tolist())
    (out_dir / "predictions.json.gz").write_bytes(gzip.compress(json.dumps(preds).encode()))
    artifacts = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file() and p.name not in ("model.json",))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    log(f"validation {val_acc:.4f} · test {test['accuracy']:.4f} · {train_seconds}s")
    update_model(name, version, status="ready", chosen_params=_jsonable(chosen), val_acc=val_acc, test=test,
                 selection=_jsonable(selection), train_seconds=train_seconds, artifacts=artifacts, artifact_bytes=size,
                 parameters=params_info, effort=effort,
                 trained=now(), **{k: m[k] for k in ("history", "finetune_config", "mlp_search") if k in m})


def param_count(method, obj, n_classes, n_train, ft_config=None):
    """Trainable parameters, stored numbers and the frozen encoder each classifier relies on."""
    enc = 0 if method == "tfidf_svm" else ENCODER_PARAMS
    if method == "zero_shot":
        return {"trainable": 0, "stored": n_classes * EMBED_DIM, "encoder": enc, "note": "label embeddings only"}
    if method == "knn":
        return {"trainable": 0, "stored": n_train * EMBED_DIM, "encoder": enc, "note": "keeps every training embedding"}
    if method == "logreg":
        return {"trainable": int(obj.coef_.size + obj.intercept_.size), "stored": int(obj.coef_.size + obj.intercept_.size), "encoder": enc}
    if method == "svm_rbf":
        sv = int(obj.support_vectors_.shape[0])
        k = int(obj.dual_coef_.size + obj.intercept_.size)
        return {"trainable": k, "stored": sv * EMBED_DIM + k, "encoder": enc, "note": f"{sv:,} support vectors"}
    if method == "mlp":
        t = int(sum(p.numel() for p in obj.parameters()))
        return {"trainable": t, "stored": t, "encoder": enc}
    if method == "tfidf_svm":
        t = int(obj.coef_.size + obj.intercept_.size)
        return {"trainable": t, "stored": t, "encoder": 0, "note": f"{obj.coef_.shape[1]:,} n-gram features"}
    if method == "ensemble":
        lr, svm = obj
        t = int(lr.coef_.size + lr.intercept_.size + svm.coef_.size + svm.intercept_.size)
        return {"trainable": t, "stored": t, "encoder": enc, "note": f"LogReg + TF-IDF SVM ({svm.coef_.shape[1]:,} n-gram features)"}
    if method in PLUGINS:
        k = count_params(obj)
        return {"trainable": k, "stored": k, "encoder": enc if PLUGINS[method].INPUT == "embeddings" else 0}
    if method == "finetune":
        return {"trainable": int(ft_config["trainable_params"]), "stored": int(ft_config["trainable_params"]),
                "encoder": int(ft_config["total_params"]), "note": "LoRA adapters + head; the encoder itself is changed"}
    return {}


def pick_quick(cands, score):
    return max(cands, key=score)


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, "tolist") else list(v) if isinstance(v, tuple) else str(v)))


# ------------------------------------------------------------------ reservations and jobs (shared by CLI and web app)
import fcntl
import re
import uuid
from contextlib import contextmanager

JOBS = REG / "jobs"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
STAGES = ("none", "candidate", "production", "archived")
PARAM_TYPES = {  # allowed manual settings per method (empty = choose on validation)
    "zero_shot": {"labels": str}, "knn": {"k": int}, "logreg": {"C": float}, "svm_rbf": {"C": float},
    "mlp": {"depth": int, "width": int, "dropout": float, "input_dropout": float, "weight_decay": float, "lr": float},
    "tfidf_svm": {"C": float},
    "ensemble": {"w": float, "T": float, "logreg_C": float, "tfidf_C": float},
    "finetune": {"epochs": int, "max_tokens": int, "lr": float, "rank": int, "head_C": float},
    **{k: dict(m.PARAMS) for k, m in PLUGINS.items()},
}


@contextmanager
def locked():
    REG.mkdir(parents=True, exist_ok=True)
    with open(REG / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def clean_params(method, params):
    allowed = PARAM_TYPES[method]
    out = {}
    for k, v in (params or {}).items():
        if v in (None, ""):
            continue
        if k not in allowed:
            raise ValueError(f"{method} has no setting '{k}'")
        t = allowed[k]
        if k == "depth" and not 0 <= int(v) <= 4:
            raise ValueError("depth must be 0–4")
        if t is list:
            v = [int(x) for x in (v if isinstance(v, list) else str(v).replace("→", ",").split(",")) if str(x).strip()]
            if not v or any(x < 1 or x > 8192 for x in v):
                raise ValueError("hidden layer sizes must be 1–8192")
        else:
            v = t(v)
        out[k] = v
    if method == "ensemble" and ("w" in out) != ("T" in out):
        raise ValueError("ensemble: set both w and T, or neither")
    return out


def new_job(kind, **args):
    job = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], "type": kind, "args": args,
           "status": "queued", "created": now()}
    write_json(JOBS / f"{job['id']}.json", job)
    return job


def reserve_split(dataset, title, description, train_per_class, test_per_class, val, seed):
    if dataset not in SPECS:
        raise ValueError(f"unknown dataset {dataset}")
    mode, value = val
    if mode == "official":
        if not official_val_split(dataset):
            raise ValueError(f"{dataset} has no official validation split")
        val = ("official", official_val_split(dataset))
    elif mode == "holdout_per_class":
        val = ("holdout", int(value))
    elif mode == "holdout_fraction":
        if not 0 < float(value) < 0.9:
            raise ValueError("hold-out fraction must be between 0 and 0.9")
        val = ("holdout", float(value))
    else:
        raise ValueError(f"unknown validation mode {mode}")
    for v in (train_per_class, test_per_class):
        if v is not None and not 1 <= int(v) <= 1_000_000:
            raise ValueError("per-class sizes must be positive")
    with locked():
        ensure_benchmark_split(dataset)
        taken = [p.parent.name for p in (SPLITS / dataset).glob("*/split.json")] + [p.name for p in (SPLITS / dataset).glob("s*")]
        nums = [int(t[1:]) for t in taken if re.fullmatch(r"s\d+", t)]
        split_id = f"s{max(nums, default=1) + 1}"
        config = {"train_per_class": None if train_per_class is None else int(train_per_class),
                  "test_per_class": None if test_per_class is None else int(test_per_class),
                  "val": list(val), "seed": int(seed), "title": (title or split_id)[:80]}
        write_json(split_dir(dataset, split_id) / "split.json", {
            "dataset": dataset, "id": split_id, "title": config["title"], "created": now(), "description": (description or "")[:2000],
            "config": config, "status": "queued"})
        job = new_job("build_split", dataset=dataset, split_id=split_id)
    return split_id, job


def reserve_model(name, dataset, split, method, params, description, tags=(), parent=None):
    if not NAME_RE.match(name or ""):
        raise ValueError("model name: lower-case letters, digits, - and _, up to 40 characters")
    if method not in METHODS:
        raise ValueError(f"unknown classifier {method}")
    sp = split_dir(dataset, split) / "split.json"
    if not sp.exists() or json.loads(sp.read_text()).get("status") not in ("ready", "queued", "running"):
        raise ValueError(f"split {dataset}/{split} does not exist or failed")
    params = clean_params(method, params)
    with locked():
        versions = [int(p.name[1:]) for p in (MODELS / name).glob("v*") if p.name[1:].isdigit()]
        existing = [json.loads((MODELS / name / f"v{v}" / "model.json").read_text()) for v in versions if (MODELS / name / f"v{v}" / "model.json").exists()]
        if existing and existing[0]["dataset"] != dataset:
            raise ValueError(f"model '{name}' is trained on {existing[0]['dataset']}; versions must keep the dataset (labels)")
        version = max(versions, default=0) + 1
        m = {"name": name, "version": version, "dataset": dataset, "split": split, "method": method, "params": params,
             "description": (description or "")[:4000], "tags": [str(t)[:30] for t in (tags or [])][:10], "stage": "none",
             "status": "queued", "created": now(), "parent": parent}
        write_json(model_dir(name, version) / "model.json", m)
        job = new_job("train", name=name, version=version)
    return m, job


def describe_model(name, version, description=None, tags=None, stage=None):
    fields = {}
    if description is not None:
        fields["description"] = str(description)[:4000]
    if tags is not None:
        fields["tags"] = [str(t).strip()[:30] for t in tags if str(t).strip()][:10]
    if stage is not None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        fields["stage"] = stage
    with locked():
        return update_model(name, version, **fields)


def register_external(name, dataset, method, result, artifact_dir=None, params=None, description="", tags=(),
                      effort=None, parameters=None, split="benchmark"):
    """Record a model trained outside the registry (finetune.py, gbert.py) as a new ready version of `name`."""
    with locked():
        versions = [int(p.name[1:]) for p in (MODELS / name).glob("v*") if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        m = {"name": name, "version": version, "dataset": dataset, "split": split, "method": method,
             "params": {}, "chosen_params": _jsonable(params or result.get("params") or {}),
             "description": description, "tags": list(tags), "stage": "candidate", "status": "ready",
             "created": now(), "trained": now(), "parent": None, "source": "benchmark run",
             "val_acc": result.get("val_acc"), "test": {k: v for k, v in result.items() if k in (
                 "accuracy", "macro_f1", "weighted_f1", "mcc", "balanced_accuracy", "kappa", "top3_accuracy", "tail_recall",
                 "per_class", "confusion")},
             "selection": _jsonable(result.get("selection", [])), "history": result.get("history"),
             "parameters": parameters or {}, "effort": effort or {}, "artifact_dir": rel(artifact_dir) if artifact_dir else None}
        if artifact_dir and resolve(artifact_dir).exists():
            files = [p for p in resolve(artifact_dir).rglob("*") if p.is_file()]
            m["artifacts"] = sorted(str(p.relative_to(resolve(artifact_dir))) for p in files)
            m["artifact_bytes"] = sum(p.stat().st_size for p in files)
        write_json(model_dir(name, version) / "model.json", m)
    return m


# ------------------------------------------------------------------ custom datasets (upload in a fixed format)
DATASETS = REG / "datasets"
DATASET_FORMATS = ("csv", "jsonl")


def reserve_dataset(name, source_path, fmt, title="", lang="en", domain="", description="", benchmark=False):
    """Validate an uploaded file (columns text, label, optional split = train/validation/test) and queue the build.

    Without a split column the rows are split stratified 80/10/10 (seed 13). Classes with fewer than 3 examples
    are dropped. The dataset then gets the benchmark split, embeddings and (optionally) the full benchmark."""
    import pandas as pd
    from prepare_raw import dedupe, stratified
    if not NAME_RE.match(name or "") or name in builtin_dataset_names():
        raise ValueError("dataset name: lower-case letters, digits, - and _, up to 40 characters, not an existing dataset")
    if fmt not in DATASET_FORMATS:
        raise ValueError(f"format must be one of {DATASET_FORMATS}")
    df = pd.read_csv(source_path) if fmt == "csv" else pd.read_json(source_path, lines=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("the file needs the columns 'text' and 'label' (optional: 'split')")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    counts = df["label"].value_counts()
    rare = set(counts[counts < 3].index)
    df = df[~df["label"].isin(rare)]
    if df["label"].nunique() < 2 or len(df) < 30:
        raise ValueError("need at least 2 classes and 30 usable rows")
    if "split" in df.columns:
        sp = df["split"].astype(str).str.lower().str.strip().replace({"val": "validation", "dev": "validation"})
        if not set(sp.unique()) <= {"train", "validation", "test"}:
            raise ValueError("split values must be train, validation (or val/dev) and test")
        splits = {s: dedupe(list(zip(df["text"][sp == s], df["label"][sp == s]))) for s in ("train", "validation", "test")}
        if not all(splits.values()):
            raise ValueError("every split (train, validation, test) needs rows")
        note = "Splits as provided in the uploaded file."
    else:
        splits = stratified(dedupe(list(zip(df["text"], df["label"]))))
        note = "Stratified 80/10/10 split of the uploaded file (seed 13)."
    with locked():
        out = DATASETS / name
        if (out / "dataset.json").exists():
            raise ValueError(f"dataset {name} exists already")
        import pyarrow as pa
        import pyarrow.parquet as pq
        out.mkdir(parents=True, exist_ok=True)
        for s, rows in splits.items():
            pq.write_table(pa.table({"text": [t for t, _ in rows], "label": [l for _, l in rows]}), out / f"{s}.parquet")
        meta = {"name": name, "title": (title or name)[:80], "lang": (lang or "en")[:5], "domain": (domain or "")[:60],
                "description": (description or "")[:2000], "created": now(), "status": "queued",
                "rows": {s: len(r) for s, r in splits.items()}, "classes": int(df["label"].nunique()),
                "dropped_rare_classes": sorted(rare)[:50], "n_dropped_rare_classes": len(rare),
                "split_note": f"Uploaded dataset. {note} Classes with fewer than 3 examples were dropped."}
        write_json(out / "dataset.json", meta)
        job = new_job("add_dataset", name=name, benchmark=bool(benchmark))
    return meta, job


def builtin_dataset_names():
    import data
    return {k for k, v in data.SPECS.items() if not v.get("custom")}


def build_custom_dataset(name, benchmark, log=print):
    import subprocess
    import sys
    import data
    data.load_custom_specs()
    meta_path = DATASETS / name / "dataset.json"
    meta = json.loads(meta_path.read_text())
    write_json(meta_path, {**meta, "status": "running"})
    log(f"building splits for {name}")
    m, splits = data.build(name)
    (DATA_DIR / f"{name}.json").write_text(json.dumps({"meta": m, **splits}))
    log(f"train {m['n_train']}, validation {m['n_val']}, test {m['n_test']}, {len(m['labels'])} classes")
    code = Path(__file__).resolve().parent
    for step in (["embed.py", name], *([["evaluate.py", name], ["backfill_metrics.py"]] if benchmark else [])):
        log("running " + " ".join(step))
        subprocess.run([sys.executable, *step], cwd=code, check=True)
    if benchmark:
        subprocess.run(["bash", "publish.sh"], cwd=code, check=False)
    ensure_benchmark_split(name)
    write_json(meta_path, {**meta, "status": "ready", "benchmarked": bool(benchmark)})
    log("dataset ready")


def list_jobs(limit=50):
    return [json.loads(p.read_text()) for p in sorted(JOBS.glob("*.json"), reverse=True)[:limit]]


RUNNING = {}  # job id -> Popen of the running job (for cancel)


def set_job(job_id, **fields):
    path = JOBS / f"{job_id}.json"
    with locked():
        j = json.loads(path.read_text())
        j.update(fields)
        write_json(path, j)
    return j


def mark_cancelled(job):
    a = job["args"]
    if job["type"] == "train":
        update_model(a["name"], a["version"], status="cancelled")
    elif job["type"] == "build_split":
        p = split_dir(a["dataset"], a["split_id"]) / "split.json"
        write_json(p, {**json.loads(p.read_text()), "status": "cancelled"})
    elif job["type"] == "add_dataset":
        p = DATASETS / a["name"] / "dataset.json"
        write_json(p, {**json.loads(p.read_text()), "status": "cancelled"})


def tail_log(job_id, lines=200):
    p = JOBS / f"{job_id}.log"
    if not p.exists():
        return ""
    text = p.read_text(errors="replace").replace("\r", "\n")
    kept = [l for l in text.splitlines() if l.strip() and "Warning" not in l and "warnings.warn" not in l]
    return "\n".join(kept[-lines:])


def worker(python=None, poll=3):
    """Run queued jobs one at a time, oldest first, each as `python jobs.py run <id>` (or via remote.py when
    EB_REMOTE_HOST is set, so the GPU work happens on that server). Jobs left running by a restart are failed."""
    import os
    import subprocess
    import sys
    python = python or os.getenv("EB_PYTHON") or sys.executable
    code = Path(__file__).resolve().parent
    for j in list_jobs(1000):
        if j["status"] == "running":
            set_job(j["id"], status="failed", error="interrupted by a restart", finished=now())
    while True:
        queued = sorted((j for j in list_jobs(1000) if j["status"] == "queued"), key=lambda j: j["id"])
        if not queued:
            time.sleep(poll)
            continue
        job = queued[0]
        set_job(job["id"], status="running", started=now())
        cmd = [python, "jobs.py", "run", job["id"]]
        if os.getenv("EB_REMOTE_HOST"):
            cmd = [python, "remote.py", "jobs.py", "run", job["id"]]
        with open(JOBS / f"{job['id']}.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
            RUNNING[job["id"]] = proc
            rc = proc.wait()
            RUNNING.pop(job["id"], None)
        current = json.loads((JOBS / f"{job['id']}.json").read_text())
        if current["status"] == "cancelled":
            mark_cancelled(current)
        else:
            set_job(job["id"], status="done" if rc == 0 else "failed", finished=now(),
                    **({} if rc == 0 else {"error": tail_log(job["id"], 3)}))


def run_job(job_id, log=print):
    """Execute one queued job (called by the web app's worker or `jobs.py run`)."""
    path = JOBS / f"{job_id}.json"
    job = json.loads(path.read_text())
    a = job["args"]
    try:
        if job["type"] == "build_split":
            sp = split_dir(a["dataset"], a["split_id"]) / "split.json"
            meta = json.loads(sp.read_text())
            write_json(sp, {**meta, "status": "running"})
            cfg = meta["config"]
            build_split(a["dataset"], a["split_id"], cfg, meta.get("description", ""), log)
        elif job["type"] == "train":
            update_model(a["name"], a["version"], status="running", started=now())
            train_model(a["name"], a["version"], log)
        elif job["type"] == "add_dataset":
            build_custom_dataset(a["name"], a.get("benchmark", False), log)
        else:
            raise ValueError(f"unknown job type {job['type']}")
    except Exception as e:
        if job["type"] == "build_split":
            sp = split_dir(a["dataset"], a["split_id"]) / "split.json"
            write_json(sp, {**json.loads(sp.read_text()), "status": "failed", "error": str(e)})
        elif job["type"] == "add_dataset":
            mp = DATASETS / a["name"] / "dataset.json"
            write_json(mp, {**json.loads(mp.read_text()), "status": "failed", "error": str(e)})
        else:
            update_model(a["name"], a["version"], status="failed", error=str(e))
        raise

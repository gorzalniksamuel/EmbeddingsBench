"""Load registry models and predict: label, confidence, top-k and the full probability distribution.

Models are addressed as "name" (latest ready version) or "name:version". Classifiers on frozen embeddings
accept embeddings (1,024-d, from jina-embeddings-v5-text-small-classification with the "Document: " prompt) or
texts (embedded here, locally or via EB_EMBED_URL). Text classifiers (TF-IDF, ensembles, fine-tuned encoders,
gbert) need texts. Used by web/inference.py.
"""
import json
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch

import registry

EMBEDDING_METHODS = {"zero_shot", "knn", "logreg", "svm_rbf", "mlp"}
TEXT_METHODS = {"tfidf_svm", "finetune", "gbert"}
BOTH_METHODS = {"ensemble"}  # embeddings and texts
_LOCK = threading.Lock()


class ModelNotFound(LookupError):
    pass


def resolve(ref):
    """'name' or 'name:version' → model record (latest ready version by default)."""
    name, _, version = str(ref).partition(":")
    ready = [m for m in registry.list_models() if m["name"] == name and m["status"] == "ready"]
    if version:
        ready = [m for m in ready if str(m["version"]) == version.lstrip("v")]
    if not ready:
        raise ModelNotFound(f"no ready model named {ref}")
    return max(ready, key=lambda m: m["version"])


def needs(method):
    if method in EMBEDDING_METHODS:
        return "embeddings"
    if method in BOTH_METHODS:
        return "embeddings+texts"
    if method in TEXT_METHODS:
        return "texts"
    plugin = registry.PLUGINS.get(method)
    return "texts" if plugin and plugin.INPUT == "text" else "embeddings"


def label_names(m):
    meta = json.loads((registry.split_dir(m["dataset"], m["split"]) / "split.json").read_text())
    return meta["labels"], [meta["label_text"][l] for l in meta["labels"]]


def _softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def _artifact_dir(m):
    """Folder of an externally trained model; relative paths (published copies) resolve against the working root."""
    if not m.get("artifact_dir"):
        return None
    folder = Path(m["artifact_dir"])
    return folder if folder.is_absolute() else registry.ROOT / folder


class Predictor:
    def __init__(self, m):
        self.m, self.method = m, m["method"]
        self.dir = registry.model_dir(m["name"], m["version"])
        self.labels, self.label_text = label_names(m)
        self.input = needs(self.method)
        self.distribution = "probabilities"
        p, d = m.get("chosen_params") or {}, self.dir
        if self.method == "logreg":
            self.clf = joblib.load(d / "logreg.joblib")
        elif self.method == "svm_rbf":
            self.clf, self.distribution = joblib.load(d / "svm_rbf.joblib"), "softmax of SVM decision scores (not calibrated)"
        elif self.method == "zero_shot":
            self.protos = np.load(d / "label_prototypes.npy")
            self.distribution = "softmax of cosine similarities (×30, not calibrated)"
        elif self.method == "knn":
            _, data = registry.load_split(m["dataset"], m["split"])
            index = {l: i for i, l in enumerate(self.labels)}
            self.train_x = registry.embeddings(m["dataset"], [r["text"] for r in data["train"]])[0]
            self.train_y = np.array([index[r["label"]] for r in data["train"]])
            self.k = int(p.get("k", 10))
            self.distribution = "similarity-weighted neighbour votes"
        elif self.method == "mlp":
            import mlp_search
            ck = torch.load(d / "mlp.pt", map_location="cpu")
            self.net = mlp_search.build(ck["dim"], ck["n_classes"], ck["params"])
            self.net.load_state_dict(ck["state_dict"])
            self.net.eval()
        elif self.method == "tfidf_svm":
            self.parts = joblib.load(d / "tfidf_svm.joblib")
            self.distribution = "softmax of SVM decision scores (not calibrated)"
        elif self.method == "ensemble":
            self.parts = joblib.load(d / "ensemble.joblib")
        elif self.method == "finetune":
            from finetune import load_finetuned
            folder = _artifact_dir(m) or d / "finetune"
            self.classify, self.ft_model, self.ft_tok, self.ft_info = load_finetuned(folder)
        elif self.method == "gbert":
            from transformers import BertForSequenceClassification, BertTokenizer
            import gpu
            folder = _artifact_dir(m)
            self.tok = BertTokenizer.from_pretrained(folder)
            self.net = BertForSequenceClassification.from_pretrained(folder).to(gpu.device()).eval()
        elif self.method in registry.PLUGINS:
            plugin = registry.PLUGINS[self.method]
            self.plugin = plugin
            self.obj = plugin.load(d / "model") if hasattr(plugin, "load") else joblib.load(d / f"{self.method}.joblib")
        else:
            raise ValueError(f"no inference support for {self.method}")

    def proba(self, x=None, texts=None):
        n = len(texts) if x is None else len(x)
        if self.method == "logreg":
            return self.clf.predict_proba(x)
        if self.method == "svm_rbf":
            return _softmax(self.clf.decision_function(x))
        if self.method == "zero_shot":
            return _softmax(30.0 * (x @ self.protos.T))
        if self.method == "knn":
            sims = x @ self.train_x.T
            top = np.argsort(-sims, axis=1)[:, :self.k]
            votes = np.zeros((n, len(self.labels)))
            np.add.at(votes, (np.repeat(np.arange(n)[:, None], self.k, 1), self.train_y[top]), np.take_along_axis(sims, top, 1))
            return votes / np.maximum(votes.sum(1, keepdims=True), 1e-12)
        if self.method == "mlp":
            with torch.no_grad():
                return torch.softmax(self.net(torch.tensor(x, dtype=torch.float32)), 1).numpy()
        if self.method == "tfidf_svm":
            return _softmax(self.parts["svm"].decision_function(self.parts["vectorizer"].transform(texts)))
        if self.method == "ensemble":
            p = self.parts
            return p["w"] * p["logreg"].predict_proba(x) + (1 - p["w"]) * _softmax(p["T"] * p["svm"].decision_function(p["vectorizer"].transform(texts)))
        if self.method == "finetune":
            from finetune import encode_texts
            import gpu
            info, tok, out = self.ft_info, self.ft_tok, []
            for i in range(0, n, 16):
                enc = encode_texts(tok, [info["prompt"] + t for t in texts[i:i + 16]], info["max_tokens"], info.get("truncation", "head"))
                b = tok.pad({"input_ids": enc}, return_tensors="pt").to(gpu.device())
                with torch.no_grad(), gpu.autocast():
                    out.append(torch.softmax(self.ft_model(b["input_ids"], b["attention_mask"]).float(), -1).cpu().numpy())
            return np.concatenate(out)
        if self.method == "gbert":
            import gpu
            out = []
            for i in range(0, n, 16):
                b = self.tok(texts[i:i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt").to(gpu.device())
                with torch.no_grad(), gpu.autocast():
                    out.append(torch.softmax(self.net(**b).logits.float(), -1).cpu().numpy())
            return np.concatenate(out)
        from classifiers import Split
        split = Split(x=x if x is not None else np.zeros((n, 1024), np.float32), texts=texts or [""] * n,
                      y=np.zeros(n, int), n_classes=len(self.labels))
        return np.asarray(self.plugin.predict_proba(self.obj, split))


@lru_cache(maxsize=8)
def _predictor(name, version):
    return Predictor(resolve(f"{name}:{version}"))


_ENCODER = None


def embed_texts(texts):
    """Embed texts with the benchmark encoder (local GPU/CPU, or EB_EMBED_URL)."""
    global _ENCODER
    from embed import MAX_TOKENS, encode, load_encoder, token_lengths
    if _ENCODER is None:
        _ENCODER = load_encoder()
    return encode(_ENCODER, texts, token_lengths(_ENCODER, texts), MAX_TOKENS)


def predict(model_ref, embeddings=None, texts=None, top_k=3, distribution=False):
    m = resolve(model_ref)
    with _LOCK:  # one prediction at a time per process (GPU memory)
        pred = _predictor(m["name"], m["version"])
        need = pred.input
        if embeddings is not None:
            x = np.asarray(embeddings, dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != 1024:
                raise ValueError(f"embeddings must be a list of 1,024-d vectors, got shape {x.shape}")
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        else:
            x = None
        if "texts" in need and not texts:
            raise ValueError(f"{m['method']} needs texts (it does not work on embeddings alone)")
        if "embeddings" in need and x is None:
            if not texts:
                raise ValueError("send embeddings or texts")
            x = embed_texts(list(texts))
        if texts is not None and x is not None and len(texts) != len(x):
            raise ValueError("embeddings and texts must have the same length")
        p = np.asarray(pred.proba(x, list(texts) if texts else None), dtype=np.float64)
    out = []
    for row in p:
        order = np.argsort(-row)
        item = {"label": pred.label_text[order[0]], "label_id": pred.labels[order[0]], "index": int(order[0]),
                "confidence": float(row[order[0]]),
                "top_k": [{"label": pred.label_text[j], "label_id": pred.labels[j], "probability": float(row[j])} for j in order[:top_k]]}
        if distribution:
            item["distribution"] = {pred.label_text[j]: float(row[j]) for j in range(len(row))}
        out.append(item)
    return {"model": f"{m['name']}:v{m['version']}", "method": m["method"], "dataset": m["dataset"], "input": need,
            "scores": pred.distribution, "classes": len(pred.labels), "predictions": out}


def models():
    out = []
    for m in registry.list_models():
        if m["status"] == "ready":
            out.append({"model": f"{m['name']}:v{m['version']}", "name": m["name"], "version": m["version"], "method": m["method"],
                        "dataset": m["dataset"], "input": needs(m["method"]), "stage": m.get("stage"),
                        "test_accuracy": (m.get("test") or {}).get("accuracy"), "description": m.get("description", "")})
    return out

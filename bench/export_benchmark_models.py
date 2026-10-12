"""Turn the benchmark's selected classifiers into registry models (so they appear in the Models tab, can be served
by the inference API and go into the release).

For every dataset and method the classifier is refit on the benchmark training split with the setting chosen on
validation, checked against the published test accuracy, and registered as "<dataset>-<method>". Methods whose
stored state would be large are skipped above MAX_MB (e.g. TF-IDF on hundreds of classes).

usage: python export_benchmark_models.py [dataset ...]
"""
import json
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlp_search
import registry
from data import DATA_DIR
from embed import EMB_DIR, MAX_TOKENS
from evaluate import RESULTS

MAX_MB = 200
METHODS = ("zero_shot", "logreg", "mlp_tuned", "knn", "tfidf_svm", "ensemble")


def tfidf(texts, y, C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    svm = LinearSVC(C=C).fit(vec.fit_transform(texts), y)
    svm.coef_ = svm.coef_.astype(np.float32)  # halves the stored size; predictions are unchanged up to rounding
    return vec, svm


def export(name, entry, have):
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    emb = np.load(EMB_DIR / f"{name}.npz")
    labels = d["meta"]["labels"]
    index = {l: i for i, l in enumerate(labels)}
    y, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "test"))
    x, xt = emb[f"train_{MAX_TOKENS}"], emb[f"test_{MAX_TOKENS}"]
    texts = [r["text"] for r in d["train"]]
    registry.ensure_benchmark_split(name)
    for method in METHODS:
        r = entry["methods"].get(method)
        model_name = f"{name}-{method}".replace("_", "-")
        if r is None or model_name in have:
            continue
        p, files = r.get("params") or {}, {}
        n = len(labels)
        if method == "zero_shot":
            protos = emb[f"labels_{p['labels']}"]
            pred = (xt @ protos.T).argmax(1)
            files["label_prototypes.npy"] = protos
            reg_method, chosen = "zero_shot", p
        elif method == "logreg":
            clf = LogisticRegression(C=p["C"], max_iter=1000).fit(x, y)
            pred, files["logreg.joblib"], reg_method, chosen = clf.predict(xt), clf, "logreg", p
        elif method == "mlp_tuned":
            model = mlp_search.fit(p, x, y, emb[f"val_{MAX_TOKENS}"], np.array([index[r_["label"]] for r_ in d["val"]]), n)
            pred = mlp_search.proba(model, xt).argmax(1)
            files["mlp.pt"] = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "params": p, "dim": x.shape[1], "n_classes": n}
            reg_method, chosen = "mlp", p
        elif method == "knn":
            if x.nbytes / 1e6 > MAX_MB:
                print(f"  {model_name}: skipped (the training embeddings are {x.nbytes / 1e6:.0f} MB)")
                continue
            from evaluate import knn
            pred, reg_method, chosen = knn(p, x, y, xt, n), "knn", p
        elif method in ("tfidf_svm", "ensemble"):
            size_mb = 200_000 * n * 4 / 1e6
            if size_mb > MAX_MB:
                print(f"  {model_name}: skipped (~{size_mb:.0f} MB of n-gram weights)")
                continue
            C = entry["methods"]["tfidf_svm"]["params"]["C"]
            vec, svm = tfidf(texts, y, C)
            test_texts = [r_["text"] for r_ in d["test"]]
            if method == "tfidf_svm":
                pred, files["tfidf_svm.joblib"], reg_method, chosen = svm.predict(vec.transform(test_texts)), {"vectorizer": vec, "svm": svm}, "tfidf_svm", p
            else:
                from scipy.special import softmax
                lr = LogisticRegression(C=entry["methods"]["logreg"]["params"]["C"], max_iter=1000).fit(x, y)
                w, T = p["w"], p["T"]
                proba = w * lr.predict_proba(xt) + (1 - w) * softmax(T * svm.decision_function(vec.transform(test_texts)), axis=1)
                pred = proba.argmax(1)
                files["ensemble.joblib"] = {"vectorizer": vec, "svm": svm, "logreg": lr, "w": w, "T": T}
                reg_method, chosen = "ensemble", {**p, "logreg_C": entry["methods"]["logreg"]["params"]["C"], "tfidf_C": C}
        acc = accuracy_score(yt, pred)
        m = registry.register_external(model_name, name, reg_method, r, params=chosen, tags=["benchmark", method],
                                       description=f"Benchmark {method} on {d['meta']['title']}, refit with the validation-selected setting "
                                                   f"(test accuracy {acc:.4f}; published {r['accuracy']:.4f}).")
        out = registry.model_dir(m["name"], m["version"])
        for fname, obj in files.items():
            if fname.endswith(".npy"):
                np.save(out / fname, obj)
            elif fname.endswith(".pt"):
                torch.save(obj, out / fname)
            else:
                joblib.dump(obj, out / fname, compress=3)
        artifacts = sorted(p_.name for p_ in out.iterdir() if p_.is_file() and p_.name != "model.json")
        registry.update_model(m["name"], m["version"], artifacts=artifacts, artifact_bytes=sum((out / a).stat().st_size for a in artifacts),
                              refit_test_accuracy=acc)
        flag = "" if abs(acc - r["accuracy"]) < 0.005 else "  (differs from the published score)"
        print(f"  {model_name}: test {acc:.4f} vs published {r['accuracy']:.4f}{flag}", flush=True)


def main():
    results = json.loads(RESULTS.read_text())["datasets"]
    have = {m["name"] for m in registry.list_models()}
    for name in sys.argv[1:] or list(results):
        print(name, flush=True)
        export(name, results[name], have)


if __name__ == "__main__":
    main()

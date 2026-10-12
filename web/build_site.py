"""Export a static, read-only copy of the dashboard (e.g. for GitHub Pages).

Copies the page, results.json, prediction files and a registry snapshot into OUT, makes stored paths relative,
and drops raw texts of datasets whose licence does not allow redistribution (predictions and metrics stay).
Also writes results/leaderboard.md for the README.

usage: python web/build_site.py <out-dir>      (run from the repository root, with the bench/ code importable)
"""
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.getenv("EB_ROOT") or HERE.parent)
CODE = Path(os.getenv("EB_CODE_DIR") or ROOT / "bench")
sys.path.insert(0, str(CODE))

# Datasets whose test texts may be republished (open licences); the rest keep predictions only.
TEXT_REDISTRIBUTABLE = {
    "dbpedia_14": "CC BY-SA 3.0", "20_newsgroups": "public research corpus", "banking77": "CC BY 4.0",
    "clinc150": "CC BY 3.0", "ledgar": "CC BY 4.0", "medical_abstracts": "CC BY-SA 3.0",
    "gnad10": "CC BY-NC-SA 4.0", "massive_intent_de": "CC BY 4.0", "swiss_caselaw_de": "CC0", "off_de": "ODbL",
}
ABS = re.compile(r"/(?:home|root|opt|srv|mnt)/[^\"']*?/(models|registry|data|results|embeddings)/")


def relative(obj):
    """Strip machine-specific absolute paths, keeping the part from models/, registry/, data/ … on."""
    return json.loads(ABS.sub(r"\1/", json.dumps(obj)))


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "predictions").mkdir(parents=True)
    shutil.copy(HERE / "template.html", out / "index.html")
    (out / ".nojekyll").write_text("")
    results = relative(json.loads((ROOT / "results" / "results.json").read_text()))
    (out / "results.json").write_text(json.dumps(results, separators=(",", ":")))

    for name in results["datasets"]:
        src = ROOT / "results" / "predictions" / f"{name}.json.gz"
        if not src.exists():
            continue
        p = json.loads(gzip.decompress(src.read_bytes()))
        if name not in TEXT_REDISTRIBUTABLE:
            p["text"] = [""] * len(p["text"])
            p["train_snippet"] = [""] * len(p["train_snippet"])
            p["text_withheld"] = True
        (out / "predictions" / f"{name}.json").write_text(json.dumps(p, separators=(",", ":")))
        print(f"{name}: {'texts included' if name in TEXT_REDISTRIBUTABLE else 'texts withheld'}")

    # registry snapshot (read-only Models / Data tabs)
    import registry
    reg = out / "registry"
    datasets = []
    for name, spec in registry.SPECS.items():
        splits = [relative({k: v for k, v in s.items() if k != "data_path"}) for s in registry.list_splits(name)]
        datasets.append({"name": name, "title": spec["title"], "official_val": registry.official_val_split(name), "splits": splits})
    models = []
    for m in registry.list_models():
        m = relative(m)
        (reg / "models" / m["name"]).mkdir(parents=True, exist_ok=True)
        (reg / "models" / m["name"] / f"{m['version']}.json").write_text(json.dumps(m))
        light = {k: v for k, v in m.items() if k not in ("selection", "history", "mlp_search")}
        if "test" in light:
            light["test"] = {k: light["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "mcc") if k in light["test"]}
        models.append(light)
    reg.mkdir(exist_ok=True)
    (reg / "state.json").write_text(json.dumps({
        "datasets": datasets, "models": models, "jobs": [], "methods": list(registry.METHODS),
        "param_types": {k: {p: t.__name__ for p, t in v.items()} for k, v in registry.PARAM_TYPES.items()},
        "stages": list(registry.STAGES)}))

    # leaderboard for the README
    order = ["zero_shot", "knn", "logreg", "svm_rbf", "mlp_tuned", "tfidf_svm", "tfidf_char", "ensemble", "ensemble_char",
             "finetune", "ft_ensemble", "gbert"]
    short = {"zero_shot": "Zero-shot", "knn": "kNN", "logreg": "LogReg", "svm_rbf": "RBF SVM", "mlp_tuned": "MLP tuned",
             "tfidf_svm": "TF-IDF", "tfidf_char": "TF-IDF char", "ensemble": "Emb.+TF-IDF", "ensemble_char": "Emb.+char",
             "finetune": "Fine-tuned", "ft_ensemble": "FT+TF-IDF", "gbert": "gbert-large"}
    cols = [m for m in order if any(m in d["methods"] for d in results["datasets"].values())]
    lines = ["| Dataset | Lang | Classes | Train | Avg tokens | " + " | ".join(short[m] for m in cols) + " |",
             "|---|---|---:|---:|---:|" + "---:|" * len(cols)]
    for name, d in results["datasets"].items():
        accs = {m: d["methods"][m]["accuracy"] for m in cols if m in d["methods"]}
        best = max(accs.values())
        cells = [(f"**{100 * accs[m]:.1f}**" if accs[m] == best else f"{100 * accs[m]:.1f}") if m in accs else "–" for m in cols]
        L = d["meta"].get("lengths", {}).get("all", {}).get("avg_tokens")
        lines.append(f"| {d['meta']['title']} | {d['meta'].get('lang', 'en').upper()} | {len(d['meta']['labels'])} | {d['meta']['n_train']:,} | "
                     f"{round(L) if L else '–'} | " + " | ".join(cells) + " |")
    (ROOT / "results" / "leaderboard.md").write_text("\n".join(lines) + "\n")
    print("wrote", out, "and results/leaderboard.md")


if __name__ == "__main__":
    main()

"""Dataset specs: load from Hugging Face, normalise to (text, label), and build train / validation / test splits.

Protocol: models are fit on train, every hyperparameter (and MLP early stopping) is chosen on validation,
and test is touched once for the reported numbers. Validation is the official split where the dataset has
one, otherwise a stratified hold-out drawn from the official training split (disjoint from train).
"""
import hashlib
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset



def _load_dotenv(path):
    """Minimal .env reader (KEY=VALUE lines); variables already set in the environment win."""
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# Working directory for data, embeddings, results, models and the registry (EB_ROOT, default: the repository root).
ROOT = Path(os.getenv("EB_ROOT") or Path(__file__).resolve().parent.parent)
DATA_DIR = ROOT / "data"


def rel(path):
    """Path relative to ROOT when inside it (stored paths stay valid across machines); resolve with ROOT / p."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
RAW_DIR = DATA_DIR / "raw"  # prepared text/label parquet files written by prepare_raw.py


def _raw(name):
    return {"data_files": {s: str(RAW_DIR / name / f"{s}.parquet") for s in ("train", "validation", "test")}}
SEED = 13
MAX_CHARS = 16000  # enough for the 2,048-token encoder window; longer texts are cut here

NEWSGROUPS = {
    "alt.atheism": "atheism", "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows", "comp.sys.ibm.pc.hardware": "IBM PC hardware",
    "comp.sys.mac.hardware": "Mac hardware", "comp.windows.x": "X Window System",
    "misc.forsale": "items for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey", "sci.crypt": "cryptography",
    "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "space",
    "soc.religion.christian": "Christianity", "talk.politics.guns": "gun politics",
    "talk.politics.mideast": "Middle East politics", "talk.politics.misc": "politics",
    "talk.religion.misc": "religion",
}
ARXIV = {
    "math.AC": "commutative algebra", "cs.CV": "computer vision", "cs.AI": "artificial intelligence",
    "cs.SY": "systems and control", "math.GR": "group theory", "cs.CE": "computational engineering",
    "cs.PL": "programming languages", "cs.IT": "information theory", "cs.DS": "data structures and algorithms",
    "cs.NE": "neural and evolutionary computing", "math.ST": "statistics theory",
}


def _yahoo(row):
    return "\n".join(x for x in (row["question_title"], row["question_content"], row["best_answer"]) if x)


# per_class=None keeps the whole split. val: ("official", split name) or ("holdout", per_class or fraction).
SPECS = {
    "yahoo_answers": dict(
        hf=("community-datasets/yahoo_answers_topics",), label="topic", text=_yahoo,
        train_per_class=1000, test_per_class=500, val=("holdout", 200), title="Yahoo! Answers Topics",
        split_note="Stratified random sample of the official splits: 1,000 train and 200 validation texts per class "
                   "(disjoint, both from the official train split) and 500 test texts per class from the official test split."),
    "dbpedia_14": dict(
        hf=("fancyzhx/dbpedia_14",), label="label", text=lambda r: f"{r['title']}. {r['content']}",
        train_per_class=500, test_per_class=200, val=("holdout", 100), title="DBpedia-14",
        split_note="Stratified random sample of the official splits: 500 train and 100 validation texts per class "
                   "(disjoint, from the official train split) and 200 test texts per class from the official test split."),
    "20_newsgroups": dict(
        hf=("SetFit/20_newsgroups",), label="label_text", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("holdout", 0.15), title="20 Newsgroups", readable=NEWSGROUPS,
        split_note="Official by-date split (posts in test are later than train; headers, footers and quotes stripped). "
                   "15% of the official train split is held out, stratified, as validation. Empty posts removed."),
    "banking77": dict(
        hf=("mteb/banking77",), label="label_text", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("holdout", 0.15), title="Banking77",
        split_note="Official train/test split. 15% of the official train split is held out, stratified, as validation."),
    "clinc150": dict(
        hf=("clinc/clinc_oos", "plus"), label="intent", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("official", "validation"), title="CLINC150 (in-scope)", drop={"oos"},
        split_note="Official train/validation/test splits of the \"plus\" config, with the out-of-scope class removed."),
    "arxiv_11": dict(
        hf=("ccdv/arxiv-classification", "no_ref"), label="label", text=lambda r: r["text"],
        train_per_class=800, test_per_class=None, val=("official", "validation"), title="arXiv papers (11 categories)",
        readable=ARXIV,
        split_note="Official train/validation/test splits. Train is a stratified sample of up to 800 papers per class; "
                   "validation and test are complete. Paper text is used from the start (title and abstract first)."),
    # --- added 2026-10-11: more domains, task types and German ---
    "ag_news": dict(
        hf=("fancyzhx/ag_news",), label="label", text=lambda r: r["text"],
        train_per_class=2500, test_per_class=None, val=("holdout", 500), title="AG News",
        split_note="Stratified sample of the official train split: 2,500 train and 500 validation texts per class (disjoint); "
                   "the complete official test split."),
    "ledgar": dict(
        hf=("coastalcph/lex_glue", "ledgar"), label="label", text=lambda r: r["text"],
        train_per_class=200, test_per_class=None, val=("official", "validation"), title="LEDGAR contract clauses (LexGLUE)",
        split_note="Official LexGLUE splits. Train: up to 200 clauses per class (rare classes have fewer); validation and test are complete, "
                   "so they keep the natural, very skewed class distribution."),
    "medical_abstracts": dict(
        hf=("TimSchopf/medical_abstracts",), label="condition_label", text=lambda r: r["medical_abstract"],
        train_per_class=None, test_per_class=None, val=("holdout", 0.15), title="Medical abstracts",
        readable={"1": "neoplasms", "2": "digestive system diseases", "3": "nervous system diseases",
                  "4": "cardiovascular diseases", "5": "general pathological conditions"},
        split_note="Official train/test split. 15% of the official train split is held out, stratified, as validation."),
    "emotion": dict(
        hf=("dair-ai/emotion", "split"), label="label", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("official", "validation"), title="Emotion (tweets)",
        split_note="Official train/validation/test splits (\"split\" config), complete."),
    "gnad10": dict(
        hf=("community-datasets/gnad10",), label="label", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("holdout", 0.15), title="10kGNAD German news", lang="de",
        split_note="Official train/test split of 10,273 DER STANDARD news articles (from the One Million Posts corpus). 15% of train is held out, stratified, as validation."),
    "amazon_reviews_de": dict(
        hf=("mteb/AmazonReviewsClassification", "de"), label="label", text=lambda r: r["text"],
        train_per_class=2000, test_per_class=None, val=("official", "validation"), title="Amazon reviews DE (stars)", lang="de",
        readable={"0": "1 star", "1": "2 stars", "2": "3 stars", "3": "4 stars", "4": "5 stars"},
        split_note="MARC German. Train: 2,000 reviews per star rating sampled from the official train split; official validation and test complete."),
    "massive_intent_de": dict(
        hf=("mteb/MassiveIntentClassification", "de"), label="label", text=lambda r: r["text"],
        train_per_class=None, test_per_class=None, val=("official", "validation"), title="MASSIVE intents DE", lang="de",
        split_note="Official MASSIVE de-DE train/validation/test splits, complete."),
    # --- added 2026-10-11: many classes and heavy class imbalance ---
    "hupd_ipc": dict(
        hf=("parquet",), hf_kwargs=_raw("hupd_ipc"), label="label", text=lambda r: r["text"],
        train_per_class=1000, test_per_class=None, val=("official", "validation"), min_train_per_class=20,
        title="HUPD patents (IPC subclass)",
        split_note="Harvard USPTO Patent Dataset, applications filed 2007; label = IPC subclass, text = title + abstract. "
                   "Classes with fewer than 20 training applications are dropped; only head classes are capped (1,000 per class in train), "
                   "so train keeps a long tail; the official validation and test splits keep their natural distribution."),
    "off_de": dict(
        hf=("parquet",), hf_kwargs=_raw("off_de"), label="label", text=lambda r: r["text"], lang="de",
        train_per_class=1000, test_per_class=None, val=("official", "validation"),
        title="Open Food Facts DE (product categories)",
        split_note="German product names (+ generic name and brand) from Open Food Facts; label = most specific category. "
                   "Catch-all and generic parent categories (food, beverages, dairies, snacks …) dropped, duplicate texts removed, categories with at least 100 products kept; "
                   "stratified 80/10/10 split; only head classes capped (1,000 per class in train), validation and test natural."),
    "mtop_intent_de": dict(
        hf=("parquet",), hf_kwargs={"data_files": {s: f"hf://datasets/mteb/mtop_intent/de/{s}-*.parquet" for s in ("train", "validation", "test")}},
        label="label_text", text=lambda r: r["text"], lang="de",
        train_per_class=None, test_per_class=None, val=("holdout", 0.15), title="MTOP intents DE",
        split_note="Official MTOP de train and test splits with their natural long tail of rare intents. The published "
                   "validation file of mteb/mtop_intent (de) repeats the test set, so 15% of train is held out, stratified, "
                   "as validation instead."),
    "swiss_caselaw_de": dict(
        hf=("parquet",), hf_kwargs=_raw("swiss_caselaw_de"), label="label", text=lambda r: r["text"], lang="de",
        train_per_class=1000, test_per_class=None, val=("official", "validation"),
        title="Swiss Federal Supreme Court decisions DE (legal area)",
        split_note="German decisions of the Swiss Federal Supreme Court (BGer, voilaj/swiss-caselaw); label = legal area "
                   "(French/Italian labels mapped to German), areas with at least 50 decisions; text = decision, first "
                   "16,000 characters. Stratified split: validation and test are 3,000-decision samples with the natural "
                   "distribution; only head areas capped (1,000 per area in train), so train keeps the long tail."),
}

DOMAIN = {"yahoo_answers": "community Q&A", "dbpedia_14": "encyclopedia", "20_newsgroups": "forum posts",
          "banking77": "banking intents", "clinc150": "assistant intents", "arxiv_11": "scientific papers",
          "ag_news": "news", "ledgar": "legal contracts", "medical_abstracts": "medical", "emotion": "emotion (tweets)",
          "gnad10": "news", "amazon_reviews_de": "product reviews (sentiment)", "massive_intent_de": "assistant intents",
          "hupd_ipc": "patents (603-way long tail)", "off_de": "food products", "mtop_intent_de": "assistant intents (long tail)",
          "swiss_caselaw_de": "court decisions"}


CUSTOM_DIR = ROOT / "registry" / "datasets"  # uploaded datasets (registry.reserve_dataset)


def load_custom_specs():
    """Add uploaded datasets (registry/datasets/<name>/dataset.json + {train,validation,test}.parquet) to SPECS."""
    for meta_path in sorted(CUSTOM_DIR.glob("*/dataset.json")):
        m = json.loads(meta_path.read_text())
        folder = meta_path.parent
        SPECS[m["name"]] = dict(
            hf=("parquet",), hf_kwargs={"data_files": {s: str(folder / f"{s}.parquet") for s in ("train", "validation", "test")}},
            label="label", text=lambda r: r["text"], train_per_class=None, test_per_class=None,
            val=("official", "validation"), title=m["title"], lang=m.get("lang", "en"), split_note=m["split_note"], custom=True)
        DOMAIN[m["name"]] = m.get("domain", "")


load_custom_specs()


def norm_hash(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def _rows(part, spec, per_class, rng, exclude=()):
    """Stratified sample of a HF split as [{text, label}], skipping empty texts and dropped labels."""
    feature = part.features[spec["label"]]
    to_name = (lambda v: feature.names[v]) if hasattr(feature, "names") else (lambda v: str(v))
    idx_by_label = {}
    for i, v in enumerate(part[spec["label"]]):
        if i not in exclude:
            idx_by_label.setdefault(v, []).append(i)
    chosen = []
    for v in sorted(idx_by_label, key=str):
        idx = idx_by_label[v]
        rng.shuffle(idx)
        chosen.extend(idx if per_class is None else idx[: per_class * 2])  # oversample, then drop empties
    rows, used = [], []
    for i, row in zip(sorted(chosen), part.select(sorted(chosen))):
        label = to_name(row[spec["label"]])
        if label in spec.get("drop", ()):
            continue
        text = (spec["text"](row) or "").strip()[:MAX_CHARS]
        if text:
            rows.append({"text": text, "label": label, "_i": i})
    if per_class is not None:
        by_label = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(r)
        rows = []
        for label in sorted(by_label):
            items = by_label[label]
            rng.shuffle(items)
            rows.extend(items[:per_class])
    return rows


def _stratified_split(rows, frac, rng):
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    keep, held = [], []
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        n = max(1, round(len(items) * frac))
        held.extend(items[:n])
        keep.extend(items[n:])
    return keep, held


def official_val_split(name):
    """Name of the dataset's own validation split, or None."""
    return SPECS[name]["val"][1] if SPECS[name]["val"][0] == "official" else None


def describe_split(cfg):
    """Human-readable split note for a split built from overrides."""
    pc = lambda v, what: f"all of the official {what} split" if v is None else f"up to {v:,} per class from the official {what} split"
    kind, arg = cfg["val"]
    val = (f"the official {arg} split" if kind == "official" else
           f"{arg:,} per class held out from the official train split" if isinstance(arg, int) else
           f"{arg:.0%} of the training sample held out, stratified")
    return (f"Train: {pc(cfg['train_per_class'], 'train')}. Validation: {val}. Test: {pc(cfg['test_per_class'], 'test')}. "
            f"Seed {cfg['seed']}. Validation/test texts duplicating a training text are removed.")


def build(name, overrides=None):
    """Build train/val/test. overrides may set train_per_class, test_per_class, val ("official", split) |
    ("holdout", per-class int or fraction float), and seed; without overrides this is the benchmark split."""
    spec = {**SPECS[name], **(overrides or {})}
    if overrides:
        spec["split_note"] = describe_split({"seed": SEED, **spec})
    ds = load_dataset(*spec["hf"], **spec.get("hf_kwargs", {}))
    rng = random.Random(spec.get("seed", SEED))
    if spec.get("min_train_per_class"):
        part = ds["train"]
        feature = part.features[spec["label"]]
        to_name = (lambda v: feature.names[v]) if hasattr(feature, "names") else (lambda v: str(v))
        counts = {}
        for v in part[spec["label"]]:
            counts[to_name(v)] = counts.get(to_name(v), 0) + 1
        rare = {l for l, c in counts.items() if c < spec["min_train_per_class"]}
        spec["drop"] = set(spec.get("drop", ())) | rare
    kind, arg = spec["val"]
    test = _rows(ds["test"], spec, spec["test_per_class"], rng)
    if kind == "official":
        train = _rows(ds["train"], spec, spec["train_per_class"], rng)
        val = _rows(ds[arg], spec, None, rng)
    elif isinstance(arg, int):
        val = _rows(ds["train"], spec, arg, rng)
        train = _rows(ds["train"], spec, spec["train_per_class"], rng, exclude={r["_i"] for r in val})
    else:
        train, val = _stratified_split(_rows(ds["train"], spec, spec["train_per_class"], rng), arg, rng)
    # Leakage guard: drop val/test texts that duplicate a training text (whitespace/case-normalised),
    # and test texts that duplicate a validation text.
    train_hashes = {norm_hash(r["text"]) for r in train}
    n_val, n_test = len(val), len(test)
    val = [r for r in val if norm_hash(r["text"]) not in train_hashes]
    val_hashes = {norm_hash(r["text"]) for r in val}
    test = [r for r in test if norm_hash(r["text"]) not in train_hashes | val_hashes]
    # classes the training split never sees cannot be predicted: drop them from validation and test
    seen = {r["label"] for r in train}
    unseen = {"val": sum(r["label"] not in seen for r in val), "test": sum(r["label"] not in seen for r in test)}
    val = [r for r in val if r["label"] in seen]
    test = [r for r in test if r["label"] in seen]
    splits = {"train": train, "val": val, "test": test}
    for rows in splits.values():
        for r in rows:
            r.pop("_i", None)
    labels = sorted({r["label"] for r in train})
    readable = spec.get("readable") or {}
    meta = {
        "name": name, "title": spec["title"], "hf": list(spec["hf"]), "labels": labels,
        "lang": spec.get("lang", "en"), "domain": DOMAIN.get(name, ""),
        "label_text": {l: readable.get(l, str(l).replace("_", " ")) for l in labels},
        "split_note": spec["split_note"],
        **{f"n_{k}": len(v) for k, v in splits.items()},
        "dupes_removed": {"val": n_val - len(val) - unseen["val"], "test": n_test - len(test) - unseen["test"]},
        "unseen_removed": unseen,
    }
    return meta, splits


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name in SPECS:
        out = DATA_DIR / f"{name}.json"
        if out.exists():
            print("cached", out)
            continue
        meta, splits = build(name)
        out.write_text(json.dumps({"meta": meta, **splits}))
        print(name, {k: meta[f"n_{k}"] for k in splits}, len(meta["labels"]), "classes",
              "duplicates removed:", meta["dupes_removed"], flush=True)


if __name__ == "__main__":
    main()

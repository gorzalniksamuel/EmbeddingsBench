"""Prepare datasets that need cleaning before data.py can sample them: writes data/raw/<name>/{train,validation,test}.parquet
with columns text, label.

  hupd_ipc          HUPD 2007 applications; text = title + abstract; label = IPC subclass; official splits
  off_de            Open Food Facts, German product names; label = most specific category (catch-alls dropped,
                    duplicates removed, categories with >= 100 products); stratified 80/10/10
  swiss_caselaw_de  Swiss Federal Supreme Court, German decisions; label = legal area (French/Italian names mapped to
                    German, >= 50 decisions); validation and test 3,000-decision stratified samples, rest train

usage: python prepare_raw.py [name ...]
"""
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from data import RAW_DIR, SEED

OFF_CATCH_ALL = {"en:groceries", "en:food", "en:beverages", "en:plant-based-foods-and-beverages", "en:plant-based-foods",
                 "en:snacks", "en:dairies", "en:meals"}


def write(name, splits):
    out = RAW_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pq.write_table(pa.table({"text": [r[0] for r in rows], "label": [r[1] for r in rows]}), out / f"{split}.parquet")
        print(f"  {name}/{split}: {len(rows):,} rows, {len({r[1] for r in rows}):,} labels", flush=True)


def stratified(rows, seed=SEED, val=0.1, test=0.1):
    """Stratified 80/10/10 split of (text, label) rows; every class keeps at least one row in each split."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    out = {"train": [], "validation": [], "test": []}
    for label in sorted(by):
        items = by[label]
        rng.shuffle(items)
        nv, nt = max(1, round(len(items) * val)), max(1, round(len(items) * test))
        out["validation"] += items[:nv]
        out["test"] += items[nv:nv + nt]
        out["train"] += items[nv + nt:]
    return out


def dedupe(rows):
    seen, out = set(), []
    for text, label in rows:
        key = re.sub(r"\s+", " ", text.lower()).strip()
        if key not in seen:
            seen.add(key)
            out.append((text, label))
    return out


def hupd_ipc():
    ds = load_dataset("late-interacters/hupd-2007", columns=["title", "abstract", "ipcr_label_subclass"])
    splits = {}
    for split in ("train", "validation", "test"):
        rows = [(f"{r['title'].strip()}. {r['abstract'].strip()}", r["ipcr_label_subclass"])
                for r in ds[split] if r["ipcr_label_subclass"] and r["abstract"]]
        splits[split] = rows
    write("hupd_ipc", splits)


def _lang_text(value, lang="de"):
    """Open Food Facts multilingual fields: list of {lang, text} structs (or a plain string)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    for item in value:
        if item and item.get("lang") == lang and item.get("text"):
            return item["text"].strip()
    return ""


def off_de():
    path = hf_hub_download("openfoodfacts/product-database", "food.parquet", repo_type="dataset")
    pf = pq.ParquetFile(path)
    rows = []
    for batch in pf.iter_batches(columns=["product_name", "generic_name", "brands", "compared_to_category"], batch_size=50_000):
        for name, generic, brands, cat in zip(*(batch.column(c).to_pylist() for c in ("product_name", "generic_name", "brands", "compared_to_category"))):
            if not cat or cat in OFF_CATCH_ALL:
                continue
            name = _lang_text(name)
            if len(name) < 3:
                continue
            gen = _lang_text(generic)
            text = name + (f" – {gen}" if gen and gen.lower() != name.lower() else "") + (f" ({brands.strip()})" if brands and brands.strip() else "")
            rows.append((text, cat))
    print(f"  off_de: {len(rows):,} products with a German name and a specific category", flush=True)
    rows = dedupe(rows)
    counts = Counter(l for _, l in rows)
    keep = {l for l, c in counts.items() if c >= 100}
    rows = [r for r in rows if r[1] in keep]
    print(f"  off_de: {len(rows):,} after dedupe, {len(keep)} categories with >= 100 products", flush=True)
    write("off_de", stratified(rows))


SWISS_REVISION = "fbd8e58d4a2e842af71b0243e1d8efa5513d18aa"
# French/Italian labels found on German BGer decisions, mapped to the German legal areas
SWISS_LABEL_MAP = {
    "Procédure pénale": "Strafprozess", "Procedura penale": "Strafprozess", "Droit de la famille": "Familienrecht",
    "Droit des poursuites et faillites": "Schuldbetreibungs- und Konkursrecht",
    "Diritto delle esecuzioni e del fallimento": "Schuldbetreibungs- und Konkursrecht",
    "Droit pénal (en général)": "Strafrecht (allgemein)", "Infractions": "Straftaten",
    "Assurance-maladie": "Krankenversicherung", "Assicurazione contro le malattie": "Krankenversicherung",
    "Assurance militaire": "Militärversicherung", "Droit des obligations (en général)": "Obligationenrecht (allgemein)",
    "Fonction publique": "Öffentliches Dienstverhältnis", "Assurance-chômage": "Arbeitslosenversicherung",
}


def stratified_capped(rows, seed=SEED, frac=0.1, cap=3000):
    """Stratified split with validation and test each min(frac, cap/N) of the data, natural class proportions."""
    f = min(frac, cap / len(rows))
    return stratified(rows, seed, val=f, test=f)


def swiss_caselaw_de():
    path = hf_hub_download("voilaj/swiss-caselaw", "data/bger.parquet", repo_type="dataset", revision=SWISS_REVISION)
    t = pq.read_table(path, columns=["language", "legal_area", "full_text"]).to_pylist()
    rows = [((r["full_text"] or "").strip()[:16000], SWISS_LABEL_MAP.get(r["legal_area"], r["legal_area"]))
            for r in t if r["language"] == "de" and r["legal_area"] and r["full_text"]]
    rows = dedupe([r for r in rows if len(r[0]) > 200])
    counts = Counter(l for _, l in rows)
    keep = {l for l, c in counts.items() if c >= 50}
    rows = [r for r in rows if r[1] in keep]
    print(f"  swiss_caselaw_de: {len(rows):,} decisions, {len(keep)} legal areas with >= 50 decisions", flush=True)
    write("swiss_caselaw_de", stratified_capped(rows))


PREPARERS = {"hupd_ipc": hupd_ipc, "off_de": off_de, "swiss_caselaw_de": swiss_caselaw_de}

if __name__ == "__main__":
    for name in sys.argv[1:] or PREPARERS:
        print(name, flush=True)
        PREPARERS[name]()

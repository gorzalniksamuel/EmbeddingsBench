"""Package trained models for a GitHub release: one archive per fine-tuned model (LoRA adapter + head, or the full
gbert-large model), one archive with all registry models, and a markdown table for models/README.md.
Absolute paths inside the stored metadata are made relative.

usage: python package_models.py <out-dir>      writes <out-dir>/*.tar.gz and <out-dir>/MODELS.md
"""
import io
import json
import re
import sys
import tarfile
from pathlib import Path

from data import DATA_DIR
from evaluate import RESULTS

ROOT = DATA_DIR.parent
ABS = re.compile(r"/(?:home|root|opt|srv|mnt)/[^\"'\s]*?/(models|registry|data|results|embeddings)/")
LICENCE = {"gbert": "MIT base model (deepset/gbert-large); released for non-commercial use with the datasets' terms",
           "default": "CC BY-NC 4.0 (base model jinaai/jina-embeddings-v5-text-small-classification)"}
VARIANTS = {"": "finetune", "-1024": "finetune_1024", "-headtail": "finetune_headtail", "-weighted": "finetune_weighted",
            "-la": "finetune_la", "-gbert": "gbert"}


def split_name(folder_name, datasets):
    """'gnad10-headtail' → ('gnad10', '-headtail'); 'arxiv_11' → ('arxiv_11', '')."""
    for sfx in sorted((v for v in VARIANTS if v), key=len, reverse=True):
        if folder_name.endswith(sfx) and folder_name[: -len(sfx)] in datasets:
            return folder_name[: -len(sfx)], sfx
    return (folder_name, "") if folder_name in datasets else (None, None)


def relative(text):
    return ABS.sub(r"\1/", text)


def add_tree(tar, folder, arcname):
    """Add a folder; JSON files get their absolute paths made relative."""
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.endswith((".tmp", ".lock")):
            continue
        name = f"{arcname}/{path.relative_to(folder)}"
        if path.suffix == ".json":
            data = relative(path.read_text()).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        else:
            tar.add(path, arcname=name)


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text())["datasets"]
    rows = []
    for folder in sorted((ROOT / "models").iterdir()):
        trained = (folder / "classifier.json").exists() or (folder / "config.json").exists()
        if not folder.is_dir() or folder.name.endswith("-smoke") or not trained:
            continue
        dataset, suffix = split_name(folder.name, results)
        if dataset is None:
            print("skipping", folder.name, "(no matching results)")
            continue
        method = VARIANTS[suffix]
        archive = out / f"finetuned-{folder.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            add_tree(tar, folder, folder.name)
        r = results[dataset]["methods"].get(method, {})
        rows.append((results[dataset]["meta"]["title"], method, r.get("accuracy"), r.get("macro_f1"), r.get("mcc"), archive.name,
                     archive.stat().st_size, LICENCE["gbert" if method == "gbert" else "default"]))
        print(f"{archive.name}: {archive.stat().st_size / 1e6:.0f} MB")

    reg = ROOT / "registry" / "models"
    if reg.exists() and any(reg.iterdir()):
        archive = out / "registry-models.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            add_tree(tar, reg, "registry/models")
        print(f"{archive.name}: {archive.stat().st_size / 1e6:.1f} MB")

    lines = ["| Dataset | Model | Test acc. | Macro-F1 | MCC | Release file | Size | Licence |", "|---|---|---:|---:|---:|---|---:|---|"]
    for title, method, acc, f1, mcc, name, size, lic in rows:
        fmt = lambda v, pct=True: "–" if v is None else (f"{100 * v:.1f}" if pct else f"{v:.3f}")
        lines.append(f"| {title} | {method} | {fmt(acc)} | {fmt(f1)} | {fmt(mcc, False)} | `{name}` | {size / 1e6:.0f} MB | {lic} |")
    (out / "MODELS.md").write_text("\n".join(lines) + "\n")
    print("wrote", out / "MODELS.md")


if __name__ == "__main__":
    main()

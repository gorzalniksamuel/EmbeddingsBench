"""Register fine-tuned models from benchmark runs (finetune.py, gbert.py) in the model registry, so they appear in
the Models tab like registry-trained models. Called at the end of each run; `register_all()` backfills older runs."""
import json

from data import DATA_DIR, SPECS, rel
from evaluate import RESULTS
from registry import MODELS, list_models, register_external

ROOT = DATA_DIR.parent
SOURCES = {  # results.json method → (model folder suffix, registry name suffix, description)
    "finetune": ("", "finetune", "LoRA fine-tuned encoder from the benchmark run (finetune.py), first 512 tokens."),
    "finetune_1024": ("-1024", "finetune-1024", "LoRA fine-tuned encoder, texts up to 1,024 tokens (benchmark run)."),
    "finetune_headtail": ("-headtail", "finetune-headtail", "LoRA fine-tuned encoder, head + tail truncation (benchmark run)."),
    "finetune_weighted": ("-weighted", "finetune-weighted", "LoRA fine-tuned encoder with class-weighted loss (benchmark run)."),
    "finetune_la": ("-la", "finetune-la", "LoRA fine-tuned encoder with logit-adjusted loss (benchmark run)."),
    "gbert": ("-gbert", "gbert", "deepset/gbert-large, fully fine-tuned (benchmark run, gbert.py)."),
}


def register(dataset, method, result):
    folder_suffix, name_suffix, description = SOURCES[method]
    folder = ROOT / "models" / f"{dataset}{folder_suffix}"
    name = f"{dataset}-{name_suffix}".replace("_", "-")
    cfg = result.get("config", {})
    params = {"trainable": cfg.get("trainable_params") or cfg.get("params"), "stored": cfg.get("trainable_params") or cfg.get("params"),
              "encoder": cfg.get("total_params", 0) if method != "gbert" else 0}
    effort = {"device": "GPU", "train_seconds": cfg.get("train_seconds"), "peak_vram_gib": cfg.get("peak_vram_gib")}
    return register_external(name, dataset, "gbert" if method == "gbert" else "finetune", result, rel(folder) if folder.exists() else None,
                             params=result.get("params"), description=description, tags=["benchmark", method],
                             effort=effort, parameters=params)


def register_all():
    results = json.loads(RESULTS.read_text())
    have = {(m["dataset"], m.get("tags", [None, None])[-1]) for m in list_models() if m.get("source") == "benchmark run"}
    out = []
    for dataset, entry in results["datasets"].items():
        for method in SOURCES:
            if method in entry["methods"] and (dataset, method) not in have:
                out.append(register(dataset, method, entry["methods"][method]))
    return out

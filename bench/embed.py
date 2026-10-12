"""Embed every dataset with jina-embeddings-v5-text-small-classification on the GPU, cached as .npz.

Two input windows are stored: 2,048 tokens (main) and 512 tokens (ablation). Texts that fit in 512 tokens
get identical vectors in both, so only the longer ones are encoded twice.

With EB_EMBED_URL set, texts are sent to a remote embedding service (web/inference.py on a GPU server, or any
OpenAI-compatible /v1/embeddings endpoint serving the same model) instead of loading the model locally.

usage: python embed.py [dataset ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import gpu

from data import DATA_DIR, SPECS

MODEL_ID = "jinaai/jina-embeddings-v5-text-small-classification"
REVISION = "4447914a9b5b2fb00db3ce0884602b47a08f9458"
EMB_DIR = DATA_DIR.parent / "embeddings"
MAX_TOKENS = 2048
ABLATION_TOKENS = 512
MIN_FREE_GB = float(os.getenv("EB_MIN_FREE_VRAM_GB", "4.0"))  # shared GPU: wait until other services leave room
PROMPT = "Document: "
# Zero-shot label prototypes; the variant is chosen per dataset on the validation split.
LABEL_VARIANTS = {"name": "{}", "sentence": "This text is about {}."}
SPLITS = ("train", "val", "test")


EMBED_URL = os.getenv("EB_EMBED_URL", "").rstrip("/")  # remote embedding service; empty = local model


class RemoteEncoder:
    """Client for a remote /v1/embeddings service (same model); only the tokenizer is loaded locally."""

    def __init__(self, url, batch=64):
        from transformers import AutoTokenizer
        self.url, self.batch = url, batch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("EB_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.environ['EB_API_KEY']}"

    def embed(self, texts, max_tokens):
        import urllib.request
        out = []
        for i in range(0, len(texts), self.batch):
            body = json.dumps({"model": MODEL_ID, "input": texts[i:i + self.batch], "prompt_name": "document",
                               "max_tokens": max_tokens}).encode()
            req = urllib.request.Request(f"{self.url}/v1/embeddings", data=body, headers=self.headers)
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=900) as r:
                        data = json.load(r)["data"]
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
            out.extend(e["embedding"] for e in sorted(data, key=lambda e: e["index"]))
        v = np.asarray(out, dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def wait_for_vram():
    if not EMBED_URL:
        gpu.wait_for_memory(MIN_FREE_GB)


def load_encoder():
    """The embedding model on the best available device (bf16/fp16), or a RemoteEncoder when EB_EMBED_URL is set."""
    if EMBED_URL:
        return RemoteEncoder(EMBED_URL)
    return SentenceTransformer(MODEL_ID, revision=REVISION, device=gpu.device(), model_kwargs={"dtype": gpu.amp_dtype()})


def token_lengths(model, texts, batch=512):
    """Token count per text (with the prompt), in batches so that long documents don't exhaust RAM."""
    out = []
    for i in range(0, len(texts), batch):
        out.extend(len(ids) for ids in model.tokenizer([PROMPT + t for t in texts[i:i + batch]])["input_ids"])
    return np.array(out)


def encode(model, texts, lengths, max_tokens):
    """Length-sorted batches under a padded-token budget; halves the budget on OOM."""
    if isinstance(model, RemoteEncoder):
        return model.embed(list(texts), max_tokens)
    model.max_seq_length = max_tokens
    order = np.argsort(-np.minimum(lengths, max_tokens))
    out = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    budget, pos = gpu.token_budget(), 0
    while pos < len(order):
        width = int(min(lengths[order[pos]], max_tokens))
        size = max(1, min(256, budget // max(width, 1)))
        batch = order[pos:pos + size]
        try:
            out[batch] = model.encode([texts[i] for i in batch], prompt_name="document", batch_size=len(batch),
                                      normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            pos += len(batch)
        except Exception as e:  # out of memory: halve the batch (NVML-less drivers raise an assert instead of OOM)
            if not gpu.is_oom(e):
                raise
            gpu.empty_cache()
            if budget <= max_tokens:
                raise
            budget //= 2
            print("OOM, token budget ->", budget, flush=True)
    return out


def main():
    EMB_DIR.mkdir(exist_ok=True)
    wait_for_vram()
    model = load_encoder()
    print("device:", f"remote {EMBED_URL}" if EMBED_URL else gpu.describe(), flush=True)
    for name in sys.argv[1:] or SPECS:
        out = EMB_DIR / f"{name}.npz"
        if out.exists():
            print("cached", out)
            continue
        d = json.loads((DATA_DIR / f"{name}.json").read_text())
        t0 = time.time()
        arrays = {}
        for split in SPLITS:
            texts = [r["text"] for r in d[split]]
            lengths = token_lengths(model, texts)
            arrays[f"tokens_{split}"] = lengths
            arrays[f"{split}_{MAX_TOKENS}"] = full = encode(model, texts, lengths, MAX_TOKENS)
            short = full.copy()
            long_idx = np.flatnonzero(lengths > ABLATION_TOKENS)
            if len(long_idx):
                short[long_idx] = encode(model, [texts[i] for i in long_idx], lengths[long_idx], ABLATION_TOKENS)
            arrays[f"{split}_{ABLATION_TOKENS}"] = short
            print(f"  {name}/{split}: {len(texts)} texts, {len(long_idx)} longer than {ABLATION_TOKENS} tokens, "
                  f"{np.mean(lengths > MAX_TOKENS):.1%} longer than {MAX_TOKENS}", flush=True)
        names = [d["meta"]["label_text"][l] for l in d["meta"]["labels"]]
        for key, template in LABEL_VARIANTS.items():
            label_texts = [template.format(n) for n in names]
            arrays[f"labels_{key}"] = encode(model, label_texts, token_lengths(model, label_texts), MAX_TOKENS)
        np.savez(out, **arrays)
        print(name, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

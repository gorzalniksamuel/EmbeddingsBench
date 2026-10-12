"""German-native reference: full fine-tuning of deepset/gbert-large (335M, cased WordPiece) as a sequence classifier.

Same splits and protocol as finetune.py: train on train, validation accuracy twice per epoch, the best
checkpoint is scored once on test. Inputs use the shared tokenizer-agnostic truncation (finetune.encode_texts,
[CLS] … [SEP], 512 tokens). gbert's config does not declare its model type, so the BERT classes are loaded
explicitly at a pinned revision. Results are merged as method "gbert"; models are saved to models/<dataset>-gbert/.

usage: python gbert.py <dataset> [...] [--epochs 3] [--lr 2e-5]
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer

import gpu
import store
from data import DATA_DIR, SEED, rel
from evaluate import metrics
from finetune import MODELS_DIR, batches, check_encoding, encode_texts

MODEL = "deepset/gbert-large"
REVISION = "b47b30cac9c40df3236cccb50d953df1caa19b1d"
MIN_FREE_GB = float(os.getenv("EB_GBERT_MIN_FREE_VRAM_GB", "7.0"))


def run(name, epochs, lr, batch=8, accum=2, max_tokens=512, evals_per_epoch=2):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    texts = {s: [r["text"] for r in d[s]] for s in ("train", "val", "test")}
    ys = {s: np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test")}

    tok = BertTokenizer.from_pretrained(MODEL, revision=REVISION, do_lower_case=False, strip_accents=False)
    print(f"tokenization check: {check_encoding(tok, texts['train'][:500], max_tokens)} texts identical to built-in truncation", flush=True)
    enc = {s: encode_texts(tok, texts[s], max_tokens, "head") for s in texts}
    lengths = {s: [len(e) for e in enc[s]] for s in enc}
    print(f"{sum(l >= max_tokens for l in lengths['train'])} of {len(enc['train'])} training texts cut at {max_tokens} tokens", flush=True)

    gpu.wait_for_memory(MIN_FREE_GB, timeout_s=6 * 3600)
    dev = gpu.device()
    print("device:", gpu.describe(), flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL, revision=REVISION, num_labels=n).to(dev)  # fp32 master weights
    model.gradient_checkpointing_enable()
    params = sum(p.numel() for p in model.parameters())

    def collate(split, ids):
        width = max(lengths[split][i] for i in ids)
        input_ids = torch.full((len(ids), width), tok.pad_token_id)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, i in enumerate(ids):
            input_ids[r, :lengths[split][i]] = torch.tensor(enc[split][i])
            mask[r, :lengths[split][i]] = 1
        return input_ids.to(dev), mask.to(dev)

    eval_batch = [batch * 4]

    @torch.no_grad()
    def predict(split):
        model.eval()
        proba = np.zeros((len(enc[split]), n), dtype=np.float32)
        while True:
            try:
                for ids in batches(lengths[split], eval_batch[0], rng, shuffle=False):
                    with gpu.autocast():
                        logits = model(*collate(split, ids)).logits
                    proba[ids] = torch.softmax(logits.float(), -1).cpu().numpy()
                break
            except Exception as e:
                if not gpu.is_oom(e) or eval_batch[0] == 1:
                    raise
                gpu.empty_cache()
                eval_batch[0] //= 2
        model.train()
        return proba

    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for k, p in model.named_parameters() if not any(x in k for x in no_decay)], "weight_decay": 0.01},
              {"params": [p for k, p in model.named_parameters() if any(x in k for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps_per_epoch = math.ceil(len(enc["train"]) / batch / accum)
    total = steps_per_epoch * epochs
    warmup = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup)))
    eval_every = max(1, steps_per_epoch // evals_per_epoch)
    save_dir = MODELS_DIR / f"{name}-gbert"
    best, history, step, losses = (-1.0, 0), [], 0, []
    if gpu.is_cuda():
        torch.cuda.reset_peak_memory_stats()
    peak = lambda: torch.cuda.max_memory_allocated() / 2**30 if gpu.is_cuda() else 0.0
    scaler, micro = gpu.grad_scaler(), batch
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        for group in batches(lengths["train"], batch * accum, rng):  # one optimiser step per group
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for start in range(0, len(group), micro):
                        ids = group[start:start + micro]
                        with gpu.autocast():
                            logits = model(*collate("train", ids)).logits
                        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(ys["train"][ids]).to(dev)) * len(ids) / len(group)
                        scaler.scale(loss).backward()
                        step_loss += loss.item()
                    break
                except Exception as e:
                    if not gpu.is_oom(e) or micro == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    gpu.empty_cache()
                    micro //= 2
                    print(f"out of GPU memory: micro-batch -> {micro}", flush=True)
            losses.append(step_loss)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % eval_every == 0 or step == total:
                val = accuracy_score(ys["val"], predict("val").argmax(1))
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 3), "val_acc": val,
                                "train_loss": float(np.mean(losses[-eval_every:]))})
                print(f"step {step}/{total} epoch {step / steps_per_epoch:.2f} loss {history[-1]['train_loss']:.4f} val acc {val:.4f} "
                      f"({time.time() - t0:.0f}s, peak {peak():.1f} GiB)", flush=True)
                if val > best[0]:  # keep the best checkpoint on disk (RAM is scarce)
                    best = (val, step)
                    if save_dir.exists():
                        shutil.rmtree(save_dir)
                    model.save_pretrained(save_dir)
            if step >= total:
                break
    train_seconds = time.time() - t0
    peak_gib = peak()
    del model, opt
    gpu.empty_cache()
    model = BertForSequenceClassification.from_pretrained(save_dir).to(dev)
    proba = predict("test")
    pred = proba.argmax(1)
    tok.save_pretrained(save_dir)
    config = {"base_model": MODEL, "revision": REVISION, "params": params, "epochs": epochs, "lr": lr, "batch": batch * accum,
              "max_tokens": max_tokens, "best_step": best[1], "steps": step, "train_seconds": round(train_seconds),
              "peak_vram_gib": round(peak_gib, 2), "device": gpu.describe(), "saved_to": rel(save_dir)}
    (save_dir / "classifier.json").write_text(json.dumps({"labels": labels, "label_text": d["meta"]["label_text"], "dataset": name,
                                                          "val_acc": best[0], **config}, indent=1))
    result = {**metrics(ys["test"], pred, n), "params": {"lr": lr, "epochs": epochs, "max_tokens": max_tokens}, "val_acc": best[0],
              "selection": [{"params": {"checkpoint": h["step"]}, "val_acc": h["val_acc"]} for h in history],
              "history": history, "config": config}
    print(f"best val acc {best[0]:.4f} at step {best[1]}; TEST acc {result['accuracy']:.4f} macroF1 {result['macro_f1']:.4f} mcc {result['mcc']:.4f}", flush=True)
    top = np.argsort(-proba, axis=1)[:, :3]

    def merge(res, load):
        res["datasets"][name]["methods"]["gbert"] = result
        p = load(name)
        p["pred"]["gbert"] = pred.tolist()
        p["top3"]["gbert"] = top.tolist()
        p["top3p"]["gbert"] = np.take_along_axis(proba, top, 1).round(4).tolist()
    store.update(merge)
    try:  # make the run visible in the registry's Models tab
        from register_benchmark import register
        print("registered as", register(name, "gbert", result)["name"], flush=True)
    except Exception as e:
        print("registry registration skipped:", e, flush=True)
    del model
    gpu.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    for name in a.datasets:
        print(name, flush=True)
        run(name, a.epochs, a.lr)


if __name__ == "__main__":
    main()

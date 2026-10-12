"""Fine-tune jina-embeddings-v5-text-small-classification end to end (LoRA + linear head) on one dataset.

Same splits and protocol as evaluate.py: train on train, pick the best checkpoint by validation accuracy,
score test once. The head is initialised from a logistic regression on the frozen embeddings, so step 0
equals the frozen-encoder LogReg baseline. Results are merged into results/results.json and the explorer's
predictions file as the method "finetune".

The selected checkpoint is saved to models/<dataset>/ (LoRA adapter, head, labels, settings); load it with
load_finetuned() for inference.

usage: python finetune.py <dataset> [--epochs 3] [--max-tokens 512] [--smoke]
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoModel, AutoTokenizer

import gpu

from data import DATA_DIR, SEED, rel
from embed import EMB_DIR, MAX_TOKENS, MODEL_ID, PROMPT, REVISION, wait_for_vram
import store
from evaluate import RESULTS, metrics

MODELS_DIR = DATA_DIR.parent / "models"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Classifier(torch.nn.Module):
    """Encoder → last-token pooling → L2 normalise (as the embedding model) → linear head."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def embed(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1  # right padding: last real token
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def forward(self, input_ids, attention_mask):
        return self.head(self.embed(input_ids, attention_mask))


TAIL_TOKENS = 128  # tokens kept from the end of a text with truncation="head_tail"


def special_wrapping(tok):
    """The tokenizer's own special tokens around a single text, e.g. ([CLS], [SEP]) for BERT, ((), ()) for Qwen."""
    probe = "probe text for special tokens"
    plain = tok(probe, add_special_tokens=False)["input_ids"]
    full = tok(probe, add_special_tokens=True)["input_ids"]
    for start in range(len(full) - len(plain) + 1):
        if full[start:start + len(plain)] == plain:
            return full[:start], full[start + len(plain):]
    raise RuntimeError("could not locate the text inside the tokenizer's special-token template")


def encode_texts(tok, texts, max_tokens, truncation="head"):
    """Token ids within max_tokens, tokenizer-agnostic: content is tokenized without special tokens, cut
    (head, or head + tail), and wrapped in the tokenizer's own special tokens. Works for BPE tokenizers
    without special tokens (Qwen/jina) and for BERT-style [CLS] … [SEP] tokenizers alike."""
    if truncation not in ("head", "head_tail"):
        raise ValueError(f"unknown truncation {truncation}")
    prefix, suffix = special_wrapping(tok)
    budget = max_tokens - len(prefix) - len(suffix)
    out = []
    for start in range(0, len(texts), 256):  # in chunks: full-length ids of long documents need GBs at once
        for ids in tok(texts[start:start + 256], add_special_tokens=False)["input_ids"]:
            if len(ids) > budget:
                ids = ids[:budget] if truncation == "head" else ids[:budget - TAIL_TOKENS] + ids[-TAIL_TOKENS:]
            out.append(prefix + ids + suffix)
    return out


def check_encoding(tok, texts, max_tokens):
    """Head truncation must match the tokenizer's built-in truncation exactly (comparability with earlier runs)."""
    ours = encode_texts(tok, texts, max_tokens, "head")
    ref = tok(texts, truncation=True, max_length=max_tokens)["input_ids"]
    bad = sum(a != b for a, b in zip(ours, ref))
    if bad:
        raise RuntimeError(f"tokenization check failed for {bad} of {len(texts)} texts")
    return len(texts)


def load_finetuned(path, device=None):
    """Rebuild a saved fine-tuned classifier. Returns (classify(texts) -> [(label, prob)], model, tokenizer, info)."""
    from pathlib import Path
    from peft import PeftModel
    path = Path(path)
    info = json.loads((path / "classifier.json").read_text())
    tok = AutoTokenizer.from_pretrained(info["base_model"], revision=info["revision"])
    tok.padding_side = "right"
    device = device or gpu.device()
    enc = AutoModel.from_pretrained(info["base_model"], revision=info["revision"],
                                    dtype=gpu.amp_dtype() if device.startswith("cuda") else torch.float32)
    enc = PeftModel.from_pretrained(enc, path / "adapter")
    head = torch.nn.Linear(enc.config.hidden_size, len(info["labels"]))
    head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
    model = Classifier(enc, head).to(device).eval()

    @torch.no_grad()
    def classify(texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = encode_texts(tok, [info["prompt"] + t for t in texts[i:i + batch_size]], info["max_tokens"], info.get("truncation", "head"))
            b = tok.pad({"input_ids": enc}, return_tensors="pt").to(device)
            with gpu.autocast():
                p = torch.softmax(model(b["input_ids"], b["attention_mask"]).float(), -1).cpu()
            out += [(info["label_text"][info["labels"][j]], float(p[r, j])) for r, j in enumerate(p.argmax(1).tolist())]
        return out
    return classify, model, tok, info


def batches(lengths, batch_size, rng, shuffle=True):
    """Length-grouped batches: shuffle, sort within chunks of 32 batches, shuffle batch order (less padding)."""
    idx = list(range(len(lengths)))
    if not shuffle:
        idx.sort(key=lambda i: -lengths[i])
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(idx)
    chunk = batch_size * 32
    out = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: -lengths[i])
        out += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(out)
    return out


DEFAULTS = {"epochs": 3, "max_tokens": 512, "batch": 8, "accum": 2, "lr": 1e-4, "rank": 16, "evals_per_epoch": 4, "truncation": "head", "loss": "ce"}


def train(d, x_train, C, opts, save_dir, smoke=False, log=print):
    """LoRA fine-tune on split dict d ({"meta", "train", "val", "test"}); x_train are the frozen train embeddings
    used to initialise the head (LogReg with C). Saves the best-validation checkpoint to save_dir and returns
    {"pred", "proba", "result", "test_acc"}; result has metrics, history and config."""
    o = {**DEFAULTS, **(opts or {})}
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    labels = d["meta"]["labels"]
    n = len(labels)
    index = {l: i for i, l in enumerate(labels)}
    texts = {s: [PROMPT + r["text"] for r in d[s]] for s in ("train", "val", "test")}
    ys = {s: np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test")}

    lr0 = LogisticRegression(C=C, max_iter=1000).fit(x_train, ys["train"])
    wait_for_vram()
    device = gpu.device()
    log(f"device: {gpu.describe()}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.padding_side = "right"
    # frozen base weights in the card's half precision (bf16, or fp16 before Ampere); LoRA weights stay fp32
    enc = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=gpu.amp_dtype() if gpu.is_cuda() else torch.float32)
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc.enable_input_require_grads()
    enc = get_peft_model(enc, LoraConfig(r=o["rank"], lora_alpha=2 * o["rank"], lora_dropout=0.05, target_modules=LORA_TARGETS))
    head = torch.nn.Linear(enc.config.hidden_size, n)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(lr0.coef_))
        head.bias.copy_(torch.tensor(lr0.intercept_))
    model = Classifier(enc, head).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable {trainable / 1e6:.1f}M of {total / 1e6:.0f}M parameters")

    log(f"tokenization check: {check_encoding(tok, texts['train'][:500], o['max_tokens'])} texts identical to built-in truncation")
    tokenized = {s: encode_texts(tok, texts[s], o["max_tokens"], o["truncation"]) for s in texts}
    log(f"input: {o['truncation']} truncation to {o['max_tokens']} tokens; "
        f"{sum(len(t) >= o['max_tokens'] for t in tokenized['train'])} of {len(tokenized['train'])} training texts cut")
    lengths = {s: [len(t) for t in tokenized[s]] for s in tokenized}

    def collate(split, ids):
        seqs = [tokenized[split][i] for i in ids]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), tok.pad_token_id)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        return input_ids.to(device), mask.to(device)

    eval_batch = [o["batch"] * 4]

    @torch.no_grad()
    def predict(split):
        model.eval()
        proba = np.zeros((len(tokenized[split]), n), dtype=np.float32)
        while True:
            try:
                for ids in batches(lengths[split], eval_batch[0], rng, shuffle=False):
                    with gpu.autocast():
                        logits = model(*collate(split, ids))
                    proba[ids] = torch.softmax(logits.float(), -1).cpu().numpy()
                break
            except Exception as e:
                if not gpu.is_oom(e) or eval_batch[0] == 1:
                    raise
                gpu.empty_cache()
                eval_batch[0] //= 2
                log(f"out of GPU memory in evaluation: batch -> {eval_batch[0]}")
        model.train()
        return proba

    # class-imbalance options: weighted CE (∝ 1/frequency) or logit-adjusted loss (Menon et al., 2021: + log prior in training)
    counts = np.bincount(ys["train"], minlength=n)
    class_w = torch.tensor(counts.sum() / (n * np.maximum(counts, 1)), dtype=torch.float32, device=device) if o["loss"] == "weighted" else None
    log_prior = torch.tensor(np.log(np.maximum(counts, 1) / counts.sum()), dtype=torch.float32, device=device) if o["loss"] == "logit_adjusted" else None
    if o["loss"] not in ("ce", "weighted", "logit_adjusted"):
        raise ValueError(f"unknown loss {o['loss']}")

    peak = lambda: torch.cuda.max_memory_allocated() / 2**30 if gpu.is_cuda() else 0.0
    snapshot = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")}
    head_params = list(model.head.parameters())
    lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": o["lr"]}, {"params": head_params, "lr": o["lr"]}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tokenized["train"]) / o["batch"] / o["accum"])
    total_steps = 40 if smoke else steps_per_epoch * o["epochs"]
    warmup = max(1, int(0.06 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    eval_every = max(1, steps_per_epoch // o["evals_per_epoch"]) if not smoke else 20

    val0 = accuracy_score(ys["val"], predict("val").argmax(1))
    history = [{"step": 0, "epoch": 0.0, "val_acc": val0, "train_loss": None}]
    log(f"step 0 val acc {val0:.4f}")
    best = (val0, 0, snapshot())
    if gpu.is_cuda():
        torch.cuda.reset_peak_memory_stats()
    scaler = gpu.grad_scaler()
    micro = o["batch"]  # micro-batch size; halves on out-of-memory, the effective batch (batch × accum) stays the same
    t0, step, losses = time.time(), 0, []
    model.train()
    done = False
    for epoch in range(o["epochs"]):
        for group in batches(lengths["train"], o["batch"] * o["accum"], rng):  # one optimiser step per group
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for start in range(0, len(group), micro):
                        ids = group[start:start + micro]
                        with gpu.autocast():
                            logits = model(*collate("train", ids))
                        logits = logits.float() + (log_prior if log_prior is not None else 0)
                        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ys["train"][ids], device=device), weight=class_w) * len(ids) / len(group)
                        scaler.scale(loss).backward()
                        step_loss += loss.item()
                    break
                except Exception as e:
                    if not gpu.is_oom(e) or micro == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    gpu.empty_cache()
                    micro //= 2
                    log(f"out of GPU memory: micro-batch -> {micro} (effective batch unchanged)")
            losses.append(step_loss)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % eval_every == 0 or step == total_steps:
                val = accuracy_score(ys["val"], predict("val").argmax(1))
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 3), "val_acc": val,
                                "train_loss": float(np.mean(losses[-eval_every:]))})
                log(f"step {step}/{total_steps} epoch {step / steps_per_epoch:.2f} loss {history[-1]['train_loss']:.4f} "
                    f"val acc {val:.4f} ({time.time() - t0:.0f}s, peak {peak():.1f} GiB)")
                if val > best[0]:
                    best = (val, step, snapshot())
            if step >= total_steps:
                done = True
                break
        if done:
            break
    train_seconds = time.time() - t0
    peak_gib = peak()

    model.load_state_dict(best[2], strict=False)
    proba = predict("test")
    pred = proba.argmax(1)
    acc = accuracy_score(ys["test"], pred)
    log(f"best val acc {best[0]:.4f} at step {best[1]}; TEST acc {acc:.4f} macroF1 {f1_score(ys['test'], pred, average='macro'):.4f} "
        f"mcc {matthews_corrcoef(ys['test'], pred):.4f}")
    config = {"max_tokens": o["max_tokens"], "truncation": o["truncation"], "loss": o["loss"], "epochs": o["epochs"], "batch": o["batch"] * o["accum"],
              "micro_batch": micro, "device": gpu.describe(), "lr": o["lr"],
              "lora_rank": o["rank"], "lora_targets": LORA_TARGETS, "trainable_params": trainable, "total_params": total,
              "head_init": f"logistic regression on frozen embeddings (C={C})", "best_step": best[1],
              "steps": step, "train_seconds": round(train_seconds), "peak_vram_gib": round(peak_gib, 2), "saved_to": rel(save_dir)}
    result = {**metrics(ys["test"], pred, n), "params": {"lora_rank": o["rank"], "max_tokens": o["max_tokens"], "truncation": o["truncation"], "loss": o["loss"], "lr": o["lr"], "epochs": o["epochs"]},
              "val_acc": best[0], "selection": [{"params": {"checkpoint": h["step"]}, "val_acc": h["val_acc"]} for h in history],
              "history": history, "config": config}
    (save_dir / "adapter").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(save_dir / "adapter")
    torch.save(model.head.state_dict(), save_dir / "head.pt")
    (save_dir / "classifier.json").write_text(json.dumps({
        "base_model": MODEL_ID, "revision": REVISION, "prompt": PROMPT, "max_tokens": o["max_tokens"], "truncation": o["truncation"],
        "pooling": "last token, L2-normalised", "labels": labels, "label_text": d["meta"]["label_text"],
        "dataset": d["meta"]["name"], "val_acc": best[0], "test_acc": acc, "config": config,
        "usage": f"from finetune import load_finetuned; classify, *_ = load_finetuned('{save_dir}'); classify(['some text'])",
    }, indent=1))
    log(f"saved model to {save_dir}")
    del model, enc, opt
    gpu.empty_cache()
    return {"pred": pred, "proba": proba, "result": result, "test_acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--key", default="finetune", help="method name in results.json (e.g. finetune_1024)")
    ap.add_argument("--smoke", action="store_true", help="40 steps, reload check, no result merge")
    args = ap.parse_args()
    opts = {k: getattr(args, k) for k in DEFAULTS}

    d = json.loads((DATA_DIR / f"{args.dataset}.json").read_text())
    emb = np.load(EMB_DIR / f"{args.dataset}.npz")
    C = json.loads(RESULTS.read_text())["datasets"][args.dataset]["methods"]["logreg"]["params"]["C"]
    suffix = "" if args.key == "finetune" else "-" + args.key.removeprefix("finetune_")
    save_dir = MODELS_DIR / (args.dataset + suffix + ("-smoke" if args.smoke else ""))
    out = train(d, emb[f"train_{MAX_TOKENS}"], C, opts, save_dir, smoke=args.smoke, log=lambda m: print(m, flush=True))
    pred, proba, result = out["pred"], out["proba"], out["result"]
    if args.smoke:
        classify, *_ = load_finetuned(save_dir)
        labels = d["meta"]["labels"]
        print("reload check:", classify([r["text"] for r in d["test"][:4]]), "| saved model predicted:",
              [d["meta"]["label_text"][labels[j]] for j in pred[:4]], flush=True)
        return
    top = np.argsort(-proba, axis=1)[:, :3]

    def merge(res, load):
        res["datasets"][args.dataset]["methods"][args.key] = result
        p = load(args.dataset)
        p["pred"][args.key] = pred.tolist()
        p["top3"][args.key] = top.tolist()
        p["top3p"][args.key] = np.take_along_axis(proba, top, 1).round(4).tolist()
    store.update(merge)
    print("merged into", RESULTS, flush=True)
    try:  # make the run visible in the registry's Models tab
        from register_benchmark import register
        m = register(args.dataset, args.key, result)
        print(f"registered as {m['name']}:v{m['version']}", flush=True)
    except Exception as e:  # registration must never fail a finished training run
        print("registry registration skipped:", e, flush=True)


if __name__ == "__main__":
    main()

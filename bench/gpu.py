"""Hardware abstraction for every GPU step (embedding, fine-tuning, gbert, MLPs): works on any NVIDIA GPU, falls
back to the CPU, and adapts precision and batch sizes to the card.

  device()          "cuda:N" for the GPU with the most free memory (or EB_GPU=N), else "cpu"
  amp_dtype()       bfloat16 where supported (Ampere and newer), else float16 (older cards), float32 on CPU
  autocast()        mixed-precision context for the chosen dtype (no-op on CPU)
  grad_scaler()     GradScaler when training in float16 (needed for its small range), otherwise a no-op scaler
  free_gb()         free memory on the chosen GPU (via cudaMemGetInfo, works even when NVML is unavailable)
  wait_for_memory() block until the GPU has enough free memory (shared GPUs)
  token_budget()    padded tokens per inference batch that fit into the free memory
  is_oom(e)         True for CUDA out-of-memory errors, including the NVML assert PyTorch raises when the
                    driver and NVML library versions differ

Environment: EB_GPU (index), EB_DTYPE (bf16 | fp16 | fp32), EB_MIN_FREE_VRAM_GB, EB_TOKENS_PER_GB.
"""
import contextlib
import os
import sys
import time

import torch

_DEVICE = None


def device():
    global _DEVICE
    if _DEVICE is None:
        if not torch.cuda.is_available():
            print("no CUDA GPU found: running on the CPU (slow for embedding and fine-tuning)", file=sys.stderr, flush=True)
            _DEVICE = "cpu"
        elif os.getenv("EB_GPU"):
            _DEVICE = f"cuda:{int(os.environ['EB_GPU'])}"
        else:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            _DEVICE = f"cuda:{max(range(len(free)), key=free.__getitem__)}"
        if _DEVICE.startswith("cuda"):
            torch.cuda.set_device(_DEVICE)
    return _DEVICE


def is_cuda():
    return device().startswith("cuda")


def amp_dtype():
    forced = os.getenv("EB_DTYPE")
    if forced:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[forced]
    if not is_cuda():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast():
    dt = amp_dtype()
    if not is_cuda() or dt == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dt)


class _NoScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        pass


def grad_scaler():
    return torch.amp.GradScaler("cuda") if is_cuda() and amp_dtype() == torch.float16 else _NoScaler()


def free_gb():
    if not is_cuda():
        return float("inf")
    free, _ = torch.cuda.mem_get_info(device())
    return free / 2**30


def total_gb():
    return torch.cuda.mem_get_info(device())[1] / 2**30 if is_cuda() else float("inf")


def wait_for_memory(min_gb=None, timeout_s=3 * 3600, poll_s=30):
    min_gb = float(os.getenv("EB_MIN_FREE_VRAM_GB", "4.0")) if min_gb is None else min_gb
    if not is_cuda():
        return
    min_gb = min(min_gb, total_gb() * 0.9)  # small cards: never wait for more than they have
    start = time.time()
    while free_gb() < min_gb:
        if time.time() - start > timeout_s:
            raise SystemExit(f"not enough free GPU memory ({free_gb():.1f} of {min_gb:.1f} GiB) after {timeout_s // 60} min")
        print(f"waiting for GPU memory: {free_gb():.1f} of {min_gb:.1f} GiB free", flush=True)
        time.sleep(poll_s)


def token_budget(cap=65536, floor=1024):
    """Padded tokens per inference batch: ~EB_TOKENS_PER_GB per free GiB (measured ~5k/GiB for the 0.6B encoder)."""
    if not is_cuda():
        return 4096
    per_gb = float(os.getenv("EB_TOKENS_PER_GB", "4000"))
    return int(max(floor, min(cap, (free_gb() - 1.0) * per_gb)))


def is_oom(e):
    text = str(e)
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in text or "nvmlInit" in text


def empty_cache():
    if is_cuda():
        torch.cuda.empty_cache()


def describe():
    if not is_cuda():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(device())
    return {"device": device(), "name": p.name, "memory_gib": round(p.total_memory / 2**30, 1),
            "compute_capability": f"{p.major}.{p.minor}", "dtype": str(amp_dtype()).replace("torch.", "")}


if __name__ == "__main__":
    print(describe(), f"free {free_gb():.1f} GiB, inference token budget {token_budget()}")

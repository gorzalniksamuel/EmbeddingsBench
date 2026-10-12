"""Run a bench script on a GPU server somewhere else, over SSH.

Code and the working data are synced to the remote checkout with rsync, the script runs there (with the remote's own
.env, Python and GPU), and results, models, embeddings and the registry are synced back. Requirements: SSH key access,
rsync on both ends, and a set-up checkout on the remote (same repository, its own .venv and .env).

  EB_REMOTE_HOST     user@host of the GPU server (required)
  EB_REMOTE_DIR      path of the checkout on the remote (default: EmbeddingsBench, relative to the remote home)
  EB_REMOTE_PYTHON   interpreter on the remote, relative to EB_REMOTE_DIR (default: .venv/bin/python)
  EB_SSH_OPTS        extra ssh options, e.g. "-i ~/.ssh/gpu_key -p 2222"

usage: python remote.py <script.py> [args ...]      e.g.  python remote.py finetune.py gnad10 --epochs 3
The web app's job worker uses this automatically for every job when EB_REMOTE_HOST is set.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

# no ML imports here: the local machine may have neither a GPU nor the ML stack
ROOT = Path(os.getenv("EB_ROOT") or Path(__file__).resolve().parent.parent)

UP = ["bench/", "web/", "data/", "results/", "embeddings/", "registry/"]      # local → remote before the run
DOWN = ["results/", "models/", "embeddings/", "registry/", "data/"]          # remote → local after the run


def settings():
    host = os.getenv("EB_REMOTE_HOST")
    if not host:
        raise SystemExit("set EB_REMOTE_HOST=user@gpu-server (see remote.py)")
    return host, os.getenv("EB_REMOTE_DIR", "EmbeddingsBench"), os.getenv("EB_REMOTE_PYTHON", ".venv/bin/python"), os.getenv("EB_SSH_OPTS", "")


def rsync(src, dst, ssh_opts):
    cmd = ["rsync", "-az", "--update", "--exclude", "__pycache__", "--exclude", ".lock", "-e", f"ssh {ssh_opts}".strip(), src, dst]
    subprocess.run(cmd, check=True)


def run(script, args):
    host, rdir, rpy, ssh_opts = settings()
    ssh = ["ssh", *shlex.split(ssh_opts), host]
    subprocess.run([*ssh, f"mkdir -p {shlex.quote(rdir)}"], check=True)
    for d in UP:
        if (ROOT / d).exists():
            rsync(f"{ROOT / d}/", f"{host}:{rdir}/{d}", ssh_opts)  # trailing slash: copy the contents
    python = rpy if rpy.startswith(("/", "~")) else f"../{rpy}"
    command = f"cd {shlex.quote(rdir)}/bench && {python} {shlex.quote(script)} " + " ".join(shlex.quote(a) for a in args)
    print(f"[remote {host}] {command}", flush=True)
    code = subprocess.run([*ssh, command]).returncode  # output streams back into the local log
    for d in DOWN:
        probe = subprocess.run([*ssh, f"test -e {shlex.quote(rdir)}/{d}"])
        if probe.returncode == 0:
            (ROOT / d).mkdir(parents=True, exist_ok=True)
            rsync(f"{host}:{rdir}/{d}", f"{ROOT / d}/", ssh_opts)
    return code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(run(sys.argv[1], sys.argv[2:]))

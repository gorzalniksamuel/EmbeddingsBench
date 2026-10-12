"""Locked read-modify-write of results.json and the per-dataset prediction files (jobs run concurrently)."""
import fcntl
import gzip
import json
from contextlib import contextmanager

from evaluate import PRED_DIR, RESULTS


@contextmanager
def _lock():
    with open(RESULTS.parent / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update(fn):
    """fn(results_dict, load_predictions) mutates in place; load_predictions(name) returns a dict that is saved back."""
    with _lock():
        results = json.loads(RESULTS.read_text())
        preds = {}

        def load(name):
            if name not in preds:
                preds[name] = json.loads(gzip.decompress((PRED_DIR / f"{name}.json.gz").read_bytes()))
            return preds[name]

        fn(results, load)
        tmp = RESULTS.with_suffix(".tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(RESULTS)
        for name, p in preds.items():
            path = PRED_DIR / f"{name}.json.gz"
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(json.dumps(p, separators=(",", ":")).encode(), 6))
            tmp.replace(path)

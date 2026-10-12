"""How sure are the results, and how far can the predictions be trusted? From the stored test predictions:

- bootstrap: 95 % confidence interval of each method's test accuracy (1,000 resamples of the test set, the same
  resamples for every method), and per method whether it is statistically distinguishable from the best method
  (paired bootstrap of the accuracy difference; "tied" when the 95 % interval of the difference includes 0).
- calibration (methods with stored probabilities): expected calibration error (ECE, 15 equal-width bins), the
  reliability diagram, and selective prediction: accuracy when only the most confident x % of texts are
  classified automatically (the rest go to manual review), the area under the risk-coverage curve, and the largest
  share of texts that can be automated at 95 % / 99 % accuracy.

Note: for SVM and TF-IDF scores and the ensembles, the "probabilities" are softmaxed scores, not calibrated
probabilities; that is part of what ECE measures.

usage: python reliability.py [dataset ...]
"""
import gzip
import json
import sys

import numpy as np

import store
from evaluate import PRED_DIR, RESULTS

B = 1000
BINS = 15
COVERAGES = (0.5, 0.7, 0.8, 0.9, 1.0)
TARGETS = (0.95, 0.99)
SKIP = {"zero_shot"}  # label-similarity baseline: not a candidate for "best"


def bootstrap(y, preds, rng):
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    correct = {m: (np.asarray(p) == y) for m, p in preds.items()}
    boot = {m: c[idx].mean(1) for m, c in correct.items()}
    acc = {m: float(c.mean()) for m, c in correct.items()}
    best = max((m for m in acc if m not in SKIP), key=acc.get)
    out = {}
    for m, b in boot.items():
        diff = boot[best] - b
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out[m] = {"accuracy": acc[m], "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                  "diff_to_best": float(acc[best] - acc[m]), "diff_ci": [float(lo), float(hi)],
                  "tied_with_best": bool(lo <= 0.0)}
    return {"best": best, "resamples": B, "methods": out}


def calibration(y, pred, conf):
    correct = (np.asarray(pred) == y).astype(float)
    conf = np.clip(np.asarray(conf, dtype=float), 0, 1)
    edges = np.linspace(0, 1, BINS + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
    bins, ece = [], 0.0
    for b in range(BINS):
        mask = which == b
        if mask.any():
            a, c, k = correct[mask].mean(), conf[mask].mean(), int(mask.sum())
            ece += k / len(y) * abs(a - c)
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "accuracy": float(a), "confidence": float(c), "count": k})
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    at = {f"{c:.1f}": float(cum[max(int(round(c * len(y))) - 1, 0)]) for c in COVERAGES}
    automate = {}
    for t in TARGETS:
        ok = np.nonzero(cum >= t)[0]
        k = int(ok.max()) + 1 if len(ok) else 0
        automate[f"{t:.2f}"] = {"coverage": k / len(y), "threshold": float(conf[order][k - 1]) if k else None}
    return {"ece": float(ece), "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()), "bins": bins,
            "accuracy_at_coverage": at, "aurc": float(np.mean(1 - cum)), "automate_at": automate}


def run(name, rng):
    p = json.loads(gzip.decompress((PRED_DIR / f"{name}.json.gz").read_bytes()))
    y = np.asarray(p["y"])
    preds = {m: v for m, v in p["pred"].items() if len(v) == len(y)}
    out = {"bootstrap": bootstrap(y, preds, rng), "calibration": {}}
    for m, top in p.get("top3p", {}).items():
        if m in preds and len(top) == len(y):
            out["calibration"][m] = calibration(y, preds[m], [t[0] for t in top])
    bs = out["bootstrap"]
    tied = [m for m, v in bs["methods"].items() if v["tied_with_best"] and m != bs["best"]]
    print(f"  best {bs['best']} {100 * bs['methods'][bs['best']]['accuracy']:.1f} "
          f"[{100 * bs['methods'][bs['best']]['ci'][0]:.1f}, {100 * bs['methods'][bs['best']]['ci'][1]:.1f}]; tied: {', '.join(tied) or '–'}", flush=True)
    for m, c in out["calibration"].items():
        print(f"  {m:18s} ECE {100 * c['ece']:4.1f}  acc@80% {100 * c['accuracy_at_coverage']['0.8']:.1f}  "
              f"automate@95% {100 * c['automate_at']['0.95']['coverage']:.0f}%", flush=True)

    def merge(res, load):
        res["datasets"][name]["reliability"] = out
    store.update(merge)


def main():
    results = json.loads(RESULTS.read_text())
    rng = np.random.default_rng(0)
    for name in sys.argv[1:] or list(results["datasets"]):
        if (PRED_DIR / f"{name}.json.gz").exists():
            print(name, flush=True)
            run(name, rng)


if __name__ == "__main__":
    main()

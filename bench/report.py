"""Figures and tables for the findings report (report/REPORT.md) from results/results.json.

Writes report/figures/*.png and report/tables.md (markdown tables with the numbers the report quotes).

usage: python report.py [out-dir]      default: <repo>/report
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import ROOT  # noqa: E402
from evaluate import RESULTS  # noqa: E402

# categorical palette (fixed order, colour follows the classifier)
COLORS = {"zero_shot": "#2a78d6", "knn": "#eb6834", "logreg": "#1baf7a", "svm_rbf": "#c98a00", "mlp": "#6250d6",
          "mlp_tuned": "#9c36b5", "tfidf_svm": "#8a9099", "tfidf_char": "#495057", "ensemble": "#008300",
          "ensemble_char": "#2b8a3e", "finetune": "#d6336c", "finetune_1024": "#e64980", "finetune_headtail": "#c2255c",
          "ft_ensemble": "#a61e4d", "gbert": "#1864ab", "logreg_balanced": "#087f5b", "logit_adjusted": "#0b7285",
          "mlp_weighted": "#5f3dc4", "mlp_balanced": "#862e9c", "finetune_weighted": "#f06595", "finetune_la": "#e599f7"}
NAMES = {"zero_shot": "Zero-shot", "knn": "kNN", "logreg": "LogReg", "svm_rbf": "RBF SVM", "mlp": "MLP", "mlp_tuned": "MLP (tuned)",
         "tfidf_svm": "TF-IDF", "tfidf_char": "TF-IDF char", "ensemble": "Emb.+TF-IDF", "ensemble_char": "Emb.+char TF-IDF",
         "finetune": "Fine-tuned", "finetune_1024": "FT 1,024 tok", "finetune_headtail": "FT head+tail", "ft_ensemble": "FT+TF-IDF",
         "gbert": "gbert-large", "logreg_balanced": "LogReg weighted", "logit_adjusted": "Logit-adjusted",
         "mlp_weighted": "MLP weighted", "mlp_balanced": "MLP balanced", "finetune_weighted": "FT weighted", "finetune_la": "FT logit-adj."}
MAIN = ["zero_shot", "knn", "logreg", "svm_rbf", "mlp_tuned", "tfidf_svm", "ensemble", "finetune", "ft_ensemble"]
IMB = ["logreg", "logreg_balanced", "logit_adjusted", "mlp_tuned", "mlp_weighted", "mlp_balanced", "finetune", "finetune_weighted", "finetune_la", "tfidf_svm", "ensemble"]

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e4e9", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9aa0a8", "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def short(t):
    return t.split(" (")[0]


def heatmap(R, names, out):
    methods = [m for m in MAIN + ["tfidf_char", "ensemble_char", "gbert"] if any(m in R[n]["methods"] for n in names)]
    A = np.array([[R[n]["methods"].get(m, {}).get("accuracy", np.nan) for m in methods] for n in names])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(methods), 0.9 + 0.36 * len(names)))
    im = ax.imshow(A * 100, cmap="Blues", vmin=np.nanmin(A) * 100, vmax=100, aspect="auto")
    for i in range(len(names)):
        best = np.nanmax(A[i])
        for j in range(len(methods)):
            if not np.isnan(A[i, j]):
                ax.text(j, i, f"{100 * A[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if A[i, j] > np.nanmin(A) + 0.7 * (1 - np.nanmin(A)) else "#12151a",
                        fontweight="bold" if A[i, j] == best else "normal")
                if A[i, j] == best:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#e8590c", lw=1.8))
    ax.set_xticks(range(len(methods)), [NAMES[m] for m in methods], rotation=40, ha="right")
    ax.set_yticks(range(len(names)), [short(R[n]["meta"]["title"]) for n in names])
    ax.grid(False)
    ax.set_title("Test accuracy (%) — framed: best per dataset")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.savefig(out / "leaderboard_heatmap.png")
    plt.close(fig)


def vs_logreg(R, names, out):
    methods = [m for m in MAIN if m != "logreg"]
    deltas = {m: [100 * (R[n]["methods"][m]["accuracy"] - R[n]["methods"]["logreg"]["accuracy"]) for n in names if m in R[n]["methods"]] for m in methods}
    methods = [m for m in methods if deltas[m]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, m in enumerate(methods):
        v = deltas[m]
        ax.scatter(v, [i] * len(v), s=18, color=COLORS[m], alpha=0.55, zorder=3)
        ax.scatter([np.median(v)], [i], s=90, marker="|", color="#12151a", zorder=4)
    ax.axvline(0, color="#12151a", lw=0.8)
    ax.set_yticks(range(len(methods)), [f"{NAMES[m]} (n={len(deltas[m])})" for m in methods])
    ax.set_xlabel("test accuracy minus LogReg on frozen embeddings (points); bar = median over datasets")
    ax.set_title("Which classifier beats logistic regression?")
    fig.savefig(out / "classifiers_vs_logreg.png")
    plt.close(fig)
    return {m: (float(np.mean(deltas[m])), float(np.median(deltas[m])), sum(d > 0.5 for d in deltas[m]), len(deltas[m])) for m in methods}


def by_property(R, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (key, label, logx) in zip(axes, (("classes", "number of classes", True), ("tokens", "average tokens per text", True))):
        for m in ("zero_shot", "logreg", "tfidf_svm", "ensemble", "finetune"):
            xs, ys = [], []
            for n in names:
                if m not in R[n]["methods"]:
                    continue
                meta = R[n]["meta"]
                xs.append(len(meta["labels"]) if key == "classes" else meta.get("lengths", {}).get("all", {}).get("avg_tokens", np.nan))
                ys.append(100 * R[n]["methods"][m]["accuracy"])
            ax.scatter(xs, ys, s=22, color=COLORS[m], label=NAMES[m], alpha=0.85, zorder=3)
        ax.set_xscale("log" if logx else "linear")
        ax.set_xlabel(label)
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Accuracy by number of classes and text length", fontweight="bold", fontsize=10)
    fig.savefig(out / "accuracy_by_classes_and_length.png")
    plt.close(fig)


def learning_curves(R, names, out):
    names = [n for n in names if len(R[n].get("learning_curve") or []) >= 3]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), squeeze=False)
    for ax, n in zip(axes.flat, names):
        c = R[n]["learning_curve"]
        xs = [p["shots"] for p in c]
        ax.plot(xs, [100 * p["logreg"] for p in c], "-o", ms=3, color=COLORS["logreg"], label="LogReg")
        ax.plot(xs, [100 * p["mlp"] for p in c], "-o", ms=3, color=COLORS["mlp"], label="MLP")
        ax.axhline(100 * R[n]["methods"]["zero_shot"]["accuracy"], ls="--", color=COLORS["zero_shot"], lw=1, label="zero-shot")
        ax.set_xscale("log")
        ax.set_title(short(R[n]["meta"]["title"]), fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7)
    fig.supxlabel("labelled examples per class (log)", fontsize=8)
    fig.supylabel("test accuracy (%)", fontsize=8)
    fig.suptitle("How many labels does a classifier on frozen embeddings need?", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


def finetune_gains(R, names, out):
    rows = [(n, R[n]["methods"]) for n in names if "finetune" in R[n]["methods"]]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.45 * len(rows)))
    table = []
    for i, (n, m) in enumerate(rows):
        base = m["logreg"]["accuracy"]
        best_frozen = max(v["accuracy"] for k, v in m.items() if k in ("logreg", "knn", "mlp_tuned", "svm_rbf", "tfidf_svm", "ensemble", "ensemble_char", "tfidf_char"))
        for k, mark in (("finetune", "o"), ("finetune_1024", "^"), ("finetune_headtail", "s"), ("ft_ensemble", "D"), ("gbert", "P")):
            if k in m:
                ax.scatter([100 * (m[k]["accuracy"] - base)], [i], marker=mark, s=40, color=COLORS[k], label=NAMES[k] if i == 0 or k not in [r[0] for r in table] else None, zorder=3)
        ax.scatter([100 * (best_frozen - base)], [i], marker="|", s=200, color="#12151a", zorder=4)
        table.append((n, base, m["finetune"]["accuracy"], best_frozen, m.get("ft_ensemble", {}).get("accuracy"), m.get("gbert", {}).get("accuracy"),
                      m["finetune"].get("config", {}).get("train_seconds")))
    ax.axvline(0, color="#12151a", lw=0.8)
    ax.set_yticks(range(len(rows)), [short(R[n]["meta"]["title"]) for n, _ in rows])
    ax.set_xlabel("test accuracy minus frozen LogReg (points)\nblack bar: best classifier without GPU training")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="lower right")
    ax.set_title("What fine-tuning adds")
    fig.savefig(out / "finetune_gains.png")
    plt.close(fig)
    return table


def imbalance(R, names, out):
    rows = [n for n in names if "logreg_balanced" in R[n]["methods"]]
    if not rows:
        return []
    methods = [m for m in IMB if any(m in R[n]["methods"] for n in rows)]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3.6), sharey=False, squeeze=False)
    for ax, n in zip(axes[0], rows):
        vals = [(m, R[n]["methods"][m].get("tail_recall")) for m in methods if m in R[n]["methods"] and R[n]["methods"][m].get("tail_recall") is not None]
        ax.barh(range(len(vals)), [100 * v for _, v in vals], color=[COLORS[m] for m, _ in vals])
        ax.set_yticks(range(len(vals)), [NAMES[m] for m, _ in vals], fontsize=7)
        ax.invert_yaxis()
        ratio = R[n]["meta"].get("imbalance", {}).get("ratio")
        ax.set_title(f"{short(R[n]['meta']['title'])}" + (f"\n(train imbalance {ratio:.0f}:1)" if ratio else ""), fontsize=8)
        ax.set_xlabel("tail recall (%)")
    fig.suptitle("Rare classes: recall on the third of classes with the fewest training examples", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "imbalance_tail_recall.png")
    plt.close(fig)
    return [(n, {m: (R[n]["methods"][m]["accuracy"], R[n]["methods"][m].get("balanced_accuracy"), R[n]["methods"][m]["macro_f1"],
                     R[n]["methods"][m].get("tail_recall")) for m in methods if m in R[n]["methods"]}) for n in rows]


def tokens_and_dims(R, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = [n for n in names if "ablation_512" in R[n]]
    d = [100 * (R[n]["methods"]["logreg"]["accuracy"] - R[n]["ablation_512"]["logreg"]["accuracy"]) for n in rows]
    over = [100 * R[n]["meta"]["tokens"]["test"]["over_512"] for n in rows]
    axes[0].scatter(over, d, color=COLORS["logreg"], s=24, zorder=3)
    for x, y, n in zip(over, d, rows):
        if abs(y) > 0.25 or x > 20:
            axes[0].annotate(short(R[n]["meta"]["title"]), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(0, color="#12151a", lw=0.8)
    axes[0].set_xlabel("test texts longer than 512 tokens (%)")
    axes[0].set_ylabel("LogReg: 2,048 minus 512 tokens (points)")
    axes[0].set_title("Longer input window")
    for n in names:
        mat = R[n].get("matryoshka")
        if mat:
            full = mat[-1]["accuracy"]
            axes[1].plot([p["dims"] for p in mat], [100 * (p["accuracy"] - full) for p in mat], "-o", ms=3, lw=1, alpha=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("embedding dimensions kept (Matryoshka)")
    axes[1].set_ylabel("LogReg accuracy change vs 1,024 (points)")
    axes[1].set_title("Smaller embeddings (one line per dataset)")
    fig.savefig(out / "tokens_and_dimensions.png")
    plt.close(fig)


def costs(R, out):
    c = R.get("costs", {}).get("fits", {})
    if not c.get("embedding"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    pts = c["embedding"]["points"]
    axes[0].scatter([p["lin"] / 1e6 for p in pts], [p["seconds"] / 60 for p in pts], color=COLORS["zero_shot"], s=24, zorder=3)
    xs = np.linspace(0, max(p["lin"] for p in pts), 50)
    axes[0].plot(xs / 1e6, c["embedding"]["a"] * xs / 60, color="#12151a", lw=1)
    axes[0].set_xlabel("tokens embedded (millions)")
    axes[0].set_ylabel("GPU minutes")
    axes[0].set_title(f"Embedding: ~{1 / c['embedding']['a']:,.0f} tokens/s")
    if c.get("finetune"):
        ft = c["finetune"]["points"]
        axes[1].scatter([p["units"] / 1e6 for p in ft], [p["seconds"] / 60 for p in ft], color=COLORS["finetune"], s=24, zorder=3)
        xs = np.linspace(0, max(p["units"] for p in ft), 50)
        axes[1].plot(xs / 1e6, c["finetune"]["c"] * xs / 60, color="#12151a", lw=1)
        axes[1].set_xlabel("epochs × training tokens (millions)")
        axes[1].set_ylabel("GPU minutes")
        axes[1].set_title(f"LoRA fine-tuning: ~{c['finetune']['tokens_per_s']:,.0f} tokens/s")
    fig.suptitle("GPU effort grows with tokens, not with classes (RTX 3060, shared)", fontweight="bold", fontsize=10)
    fig.savefig(out / "gpu_scaling.png")
    plt.close(fig)


def imbalance_ratio(meta):
    counts = [c for c in (meta.get("counts", {}).get("train") or meta.get("imbalance", {}).get("train_counts") or []) if c > 0]
    return f"{max(counts) / min(counts):.0f}:1" if counts else "–"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTS.read_text())
    R = res["datasets"]
    names = sorted(R, key=lambda n: (R[n]["meta"].get("lang", "en"), len(R[n]["meta"]["labels"])))
    heatmap(R, names, figs)
    comp = vs_logreg(R, names, figs)
    by_property(R, names, figs)
    learning_curves(R, names, figs)
    ft = finetune_gains(R, names, figs)
    imb = imbalance(R, names, figs)
    tokens_and_dims(R, names, figs)
    costs(res, figs)

    pct = lambda v: "–" if v is None else f"{100 * v:.1f}"
    parts = ["## Datasets", md_table(["Dataset", "Lang", "Domain", "Classes", "Train / val / test", "Avg tokens", "Train imbalance"],
             [(R[n]["meta"]["title"], R[n]["meta"].get("lang", "en").upper(), R[n]["meta"].get("domain", ""), len(R[n]["meta"]["labels"]),
               f"{R[n]['meta']['n_train']:,} / {R[n]['meta']['n_val']:,} / {R[n]['meta']['n_test']:,}",
               round(R[n]["meta"].get("lengths", {}).get("all", {}).get("avg_tokens", 0)),
               imbalance_ratio(R[n]["meta"])) for n in names])]
    parts.append("## Classifiers against LogReg")
    parts.append(md_table(["Classifier", "Mean Δ (pts)", "Median Δ (pts)", "Better by > 0.5 pts", "Datasets"],
                          [(NAMES[m], f"{a:+.1f}", f"{b:+.1f}", w, k) for m, (a, b, w, k) in sorted(comp.items(), key=lambda kv: -kv[1][0])]))
    if ft:
        parts.append("## Fine-tuning")
        parts.append(md_table(["Dataset", "Frozen LogReg", "Fine-tuned", "Best without GPU training", "FT + TF-IDF", "gbert-large", "Train minutes"],
                              [(short(R[n]["meta"]["title"]), pct(b), pct(f), pct(bf), pct(fe), pct(g), round(s / 60) if s else "–") for n, b, f, bf, fe, g, s in ft]))
    if imb:
        parts.append("## Class imbalance (accuracy / balanced accuracy / macro-F1 / tail recall, %)")
        for n, rows in imb:
            parts.append(f"**{R[n]['meta']['title']}**\n")
            parts.append(md_table(["Classifier", "Accuracy", "Balanced acc.", "Macro-F1", "Tail recall"],
                                  [(NAMES[m], pct(a), pct(b), pct(f), pct(t)) for m, (a, b, f, t) in rows.items()]))
    (out / "tables.md").write_text("\n\n".join(parts) + "\n")
    print("wrote", figs, "and", out / "tables.md")


if __name__ == "__main__":
    main()

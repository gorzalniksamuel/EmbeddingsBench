"""Automatic bias–variance search for an MLP on frozen embeddings.

Optuna TPE search over capacity (depth 0–3, width 64–4096) and regularisation (dropout, input dropout,
weight decay, learning rate); every trial trains on train with early stopping on validation loss and is
scored on validation. Depth 0 is a linear softmax classifier, so the search can also decide that more
capacity does not help. Train accuracy is recorded next to validation accuracy for every trial, which shows
under-fitting (both low) and over-fitting (train ≫ validation).

usage: python mlp_search.py [--trials 40] [dataset ...]   merges "mlp_tuned" into results.json
"""
import argparse
import json
import time

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score

import store
from data import DATA_DIR, SEED, SPECS
from embed import EMB_DIR, MAX_TOKENS
from evaluate import DEVICE, metrics

MANUAL_DEFAULTS = {"depth": 1, "width": 512, "dropout": 0.3, "input_dropout": 0.0, "weight_decay": 1e-2, "lr": 1e-3}
SPACE = {"depth": (0, 3), "width": (64, 4096), "dropout": (0.0, 0.6), "input_dropout": (0.0, 0.3),
         "weight_decay": (1e-6, 1e-1), "lr": (1e-4, 3e-3)}


def build(dim, n_classes, p):
    layers, d = [torch.nn.Dropout(p.get("input_dropout", 0.0))], dim
    for i in range(p["depth"]):
        h = max(16, int(p["width"]) // (2 ** i))  # pyramid: width, width/2, width/4
        layers += [torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(p["dropout"])]
        d = h
    layers.append(torch.nn.Linear(d, n_classes))
    return torch.nn.Sequential(*layers)


def fit(p, x, y, xv, yv, n_classes, seed=SEED, max_epochs=300, patience=10):
    """Train with AdamW; early stopping on validation loss with the best epoch restored."""
    torch.manual_seed(seed)
    model = build(x.shape[1], n_classes, p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    xt, yt = torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)
    xvt, yvt = torch.tensor(xv, device=DEVICE), torch.tensor(yv, device=DEVICE)
    batch = min(256, max(8, len(x) // 4))
    gen = torch.Generator().manual_seed(seed)
    best, state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(batch):
            idx = idx.to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.cross_entropy(model(xvt), yvt).item()
        if vl < best - 1e-4:
            best, bad, epochs = vl, 0, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    model.epochs = epochs
    return model


def proba(model, x):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(x, device=DEVICE)), 1).cpu().numpy()


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def search(x, y, xv, yv, n_classes, n_trials=40, seed=SEED, log=print):
    """Returns (best_params, best_model, trials) with train/val accuracy and size per trial."""
    trials, models = [], {}

    def objective(trial):
        p = {"depth": trial.suggest_int("depth", *SPACE["depth"]),
             "width": trial.suggest_int("width", *SPACE["width"], log=True),
             "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
             "input_dropout": trial.suggest_float("input_dropout", *SPACE["input_dropout"]),
             "weight_decay": trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True),
             "lr": trial.suggest_float("lr", *SPACE["lr"], log=True)}
        if p["depth"] == 0:
            p["width"], p["dropout"] = 0, 0.0
        t0 = time.time()
        model = fit(p, x, y, xv, yv, n_classes, seed=seed)
        tr = accuracy_score(y, proba(model, x).argmax(1))
        va = accuracy_score(yv, proba(model, xv).argmax(1))
        trials.append({"trial": trial.number, "params": p, "n_params": n_params(model), "train_acc": tr, "val_acc": va,
                       "epochs": model.epochs, "seconds": round(time.time() - t0, 1)})
        models[trial.number] = model.cpu()
        if len(models) > 1:  # keep only the best model in memory
            keep = max(trials, key=lambda t: t["val_acc"])["trial"]
            for k in [k for k in models if k != keep]:
                models.pop(k)
        log(f"  trial {trial.number:2d} depth={p['depth']} width={p['width']:4d} drop={p['dropout']:.2f} "
            f"wd={p['weight_decay']:.1e} lr={p['lr']:.1e} params={trials[-1]['n_params']:,} train={tr:.4f} val={va:.4f}")
        return va

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    study.enqueue_trial({"depth": 0, "width": 64, "dropout": 0.0, "input_dropout": 0.0, "weight_decay": 1e-4, "lr": 1e-3})
    study.enqueue_trial({"depth": 1, "width": 512, "dropout": 0.3, "input_dropout": 0.0, "weight_decay": 1e-2, "lr": 1e-3})
    study.optimize(objective, n_trials=n_trials)
    best = max(trials, key=lambda t: t["val_acc"])
    return best["params"], models[best["trial"]].to(DEVICE), trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(SPECS))
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    for name in a.datasets:
        d = json.loads((DATA_DIR / f"{name}.json").read_text())
        emb = np.load(EMB_DIR / f"{name}.npz")
        labels = d["meta"]["labels"]
        index = {l: i for i, l in enumerate(labels)}
        y, yv, yt = (np.array([index[r["label"]] for r in d[s]]) for s in ("train", "val", "test"))
        x, xv, xt = (emb[f"{s}_{MAX_TOKENS}"] for s in ("train", "val", "test"))
        print(name, flush=True)
        t0 = time.time()
        best, model, trials = search(x, y, xv, yv, len(labels), a.trials, log=lambda m: print(m, flush=True))
        pr = proba(model, xt)
        pred = pr.argmax(1)
        best_trial = max(trials, key=lambda t: t["val_acc"])
        result = {**metrics(yt, pred, len(labels)), "params": best, "val_acc": best_trial["val_acc"],
                  "train_acc": best_trial["train_acc"], "n_params": best_trial["n_params"],
                  "selection": [{"params": t["params"], "val_acc": t["val_acc"], "epochs": t["epochs"]} for t in trials],
                  "search_seconds": round(time.time() - t0)}
        print(f"  best {best} → val {best_trial['val_acc']:.4f} test {result['accuracy']:.4f}", flush=True)
        top = np.argsort(-pr, axis=1)[:, :3]

        def merge(res, load, name=name, result=result, trials=trials, pred=pred, top=top, pr=pr):
            res["datasets"][name]["methods"]["mlp_tuned"] = result
            res["datasets"][name]["mlp_search"] = {"space": SPACE, "trials": trials}
            p = load(name)
            p["pred"]["mlp_tuned"] = pred.tolist()
            p["top3"]["mlp_tuned"] = top.tolist()
            p["top3p"]["mlp_tuned"] = np.take_along_axis(pr, top, 1).round(4).tolist()
        store.update(merge)


if __name__ == "__main__":
    main()

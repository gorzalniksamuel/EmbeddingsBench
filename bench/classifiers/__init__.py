"""Classifier plugins: drop a module into this folder to add a new model type to the registry, CLI and dashboard.

A plugin module defines

    NAME        unique id, lower-case (e.g. "nearest_centroid")
    LABEL       display name
    DESCRIPTION one or two sentences for the dashboard
    INPUT       "embeddings" (frozen 1,024-d embeddings) or "text" (raw texts)
    PARAMS      {setting: type} that can be set manually, e.g. {"C": float}
    GRID        list of settings tried on validation when none are given
    fit(params, train) -> model                 train: Split
    predict_proba(model, split) -> (n, classes)  probabilities (or one-hot scores)

and optionally save(model, path) / load(path) (default: joblib). Split has .x (embeddings, n × 1024 float32),
.texts (list of str), .y (int labels) and .n_classes. The registry picks the GRID entry with the best
validation accuracy, scores test once and stores metrics, parameters and artifacts like any other model.
"""
import importlib
import pkgutil
from dataclasses import dataclass

import numpy as np

REQUIRED = ("NAME", "LABEL", "DESCRIPTION", "INPUT", "PARAMS", "GRID", "fit", "predict_proba")


@dataclass
class Split:
    x: np.ndarray
    texts: list
    y: np.ndarray
    n_classes: int


def discover():
    """{name: module} for every valid plugin in this package."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        missing = [k for k in REQUIRED if not hasattr(mod, k)]
        if missing:
            raise ImportError(f"classifier plugin {info.name} misses {missing}")
        if mod.INPUT not in ("embeddings", "text"):
            raise ImportError(f"classifier plugin {info.name}: INPUT must be 'embeddings' or 'text'")
        found[mod.NAME] = mod
    return found


def count_params(model):
    """Best-effort count of learned numbers (numpy arrays on the model object)."""
    total = 0
    for v in getattr(model, "__dict__", {}).values():
        if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
            total += v.size
    return total or None

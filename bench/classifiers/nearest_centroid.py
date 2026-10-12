"""Example plugin on embeddings: cosine nearest class centroid (one mean embedding per class).

Copy this file to add your own embedding-based classifier; see classifiers/__init__.py for the interface.
"""
import numpy as np

NAME = "nearest_centroid"
LABEL = "Nearest class centroid"
DESCRIPTION = "Each class is the mean of its training embeddings; a text gets the class with the most similar centroid (cosine), softmax-scaled."
INPUT = "embeddings"
PARAMS = {"temperature": float}
GRID = [{"temperature": t} for t in (10.0, 30.0, 100.0)]


class Model:
    def __init__(self, centroids, temperature):
        self.centroids = centroids
        self.temperature = temperature


def fit(params, train):
    c = np.stack([train.x[train.y == k].mean(0) if (train.y == k).any() else np.zeros(train.x.shape[1], np.float32)
                  for k in range(train.n_classes)])
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return Model(c.astype(np.float32), float(params["temperature"]))


def predict_proba(model, split):
    s = model.temperature * (split.x @ model.centroids.T)
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)

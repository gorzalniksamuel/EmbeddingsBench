"""Example plugin on raw text: word 1–2-gram TF-IDF + logistic regression (gives probabilities, unlike the SVM).

Copy this file to add your own text-based classifier; see classifiers/__init__.py for the interface.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NAME = "tfidf_logreg"
LABEL = "TF-IDF + logistic regression"
DESCRIPTION = "Word 1–2-gram TF-IDF features with a logistic regression; a probabilistic text baseline that ignores the embeddings."
INPUT = "text"
PARAMS = {"C": float}
GRID = [{"C": c} for c in (1.0, 10.0, 100.0)]


class Model:
    def __init__(self, vectorizer, clf):
        self.vectorizer, self.clf = vectorizer, clf


def fit(params, train):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000)
    clf = LogisticRegression(C=params["C"], max_iter=1000).fit(vec.fit_transform(train.texts), train.y)
    return Model(vec, clf)


def predict_proba(model, split):
    import numpy as np
    p = model.clf.predict_proba(model.vectorizer.transform(split.texts))
    out = np.zeros((len(split.texts), split.n_classes), dtype=np.float32)
    out[:, model.clf.classes_] = p  # classes absent from train get probability 0
    return out

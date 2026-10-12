## Datasets

| Dataset | Lang | Domain | Classes | Train / val / test | Avg tokens | Train imbalance |
|---|---|---|---|---|---|---|
| GermEval 2018 offensive language | DE | tweets (offensive language) | 4 | 4,258 / 751 / 3,398 | 45 | 47:1 |
| Amazon reviews DE (stars) | DE | product reviews (sentiment) | 5 | 10,000 / 5,000 / 4,999 | 65 | 1:1 |
| 10kGNAD German news | DE | news | 9 | 7,860 / 1,384 / 1,027 | 732 | 3:1 |
| Swiss Federal Supreme Court decisions DE (legal area) | DE | court decisions | 47 | 25,737 / 3,002 / 3,002 | 3413 | 19:1 |
| MASSIVE intents DE | DE | assistant intents | 60 | 11,514 / 1,943 / 2,845 | 12 | 202:1 |
| MTOP intents DE | DE | assistant intents (long tail) | 103 | 11,072 / 1,956 / 1,756 | 14 | 1175:1 |
| Open Food Facts DE (product categories) | DE | food products | 342 | 69,366 / 8,790 / 8,790 | 18 | 12:1 |
| AG News | EN | news | 4 | 10,000 / 2,000 / 7,599 | 54 | 1:1 |
| Medical abstracts | EN | medical | 5 | 9,817 / 1,221 / 1,878 | 274 | 3:1 |
| Emotion (tweets) | EN | emotion (tweets) | 6 | 16,000 / 1,995 / 1,986 | 22 | 9:1 |
| Yahoo! Answers Topics | EN | community Q&A | 10 | 10,000 / 2,000 / 5,000 | 127 | 1:1 |
| arXiv papers (11 categories) | EN | scientific papers | 11 | 8,800 / 2,446 / 2,441 | 4300 | 1:1 |
| DBpedia-14 | EN | encyclopedia | 14 | 7,000 / 1,400 / 2,800 | 79 | 1:1 |
| 20 Newsgroups | EN | forum posts | 20 | 9,361 / 1,644 / 7,299 | 275 | 2:1 |
| Banking77 | EN | banking intents | 77 | 8,497 / 1,496 / 3,069 | 15 | 5:1 |
| LEDGAR contract clauses (LexGLUE) | EN | legal contracts | 100 | 19,092 / 9,914 / 9,855 | 144 | 9:1 |
| CLINC150 (in-scope) | EN | assistant intents | 150 | 15,000 / 2,997 / 4,498 | 11 | 1:1 |
| HUPD patents (IPC subclass) | EN | patents (603-way long tail) | 408 | 98,195 / 18,934 / 18,901 | 151 | 50:1 |

## Classifiers against LogReg

| Classifier | Mean Δ (pts) | Median Δ (pts) | Better by > 0.5 pts | Datasets |
|---|---|---|---|---|
| Emb.+TF-IDF | +3.7 | +3.2 | 12 | 17 |
| Fine-tuned | +3.5 | +4.0 | 5 | 6 |
| TF-IDF | +0.5 | +0.3 | 9 | 18 |
| MLP (tuned) | +0.2 | +0.0 | 3 | 15 |
| RBF SVM | -0.9 | -0.1 | 3 | 17 |
| kNN | -6.5 | -5.3 | 0 | 18 |
| Zero-shot | -39.9 | -40.9 | 0 | 18 |

## Fine-tuning

| Dataset | Frozen LogReg | Fine-tuned | Best without GPU training | FT + TF-IDF | gbert-large | Train minutes |
|---|---|---|---|---|---|---|
| Amazon reviews DE | 62.2 | 63.6 | 63.3 | – | – | 37 |
| 10kGNAD German news | 84.1 | 88.6 | 88.7 | – | 91.3 | 76 |
| MASSIVE intents DE | 87.0 | 87.0 | 87.4 | – | – | 26 |
| Yahoo! Answers Topics | 67.1 | 71.5 | 70.8 | – | – | 39 |
| arXiv papers | 82.9 | 86.6 | 87.7 | – | – | 150 |
| 20 Newsgroups | 65.7 | 72.7 | 75.0 | – | – | 46 |

## Class imbalance (accuracy / balanced accuracy / macro-F1 / tail recall, %)

**LEDGAR contract clauses (LexGLUE)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 74.2 | 71.5 | 68.3 | 67.0 |
| LogReg weighted | 74.1 | 72.7 | 68.8 | 69.8 |
| Logit-adjusted | 74.2 | 71.5 | 68.3 | 67.0 |
| MLP (tuned) | 76.8 | 72.6 | 69.7 | 68.1 |
| MLP weighted | 76.4 | 72.7 | 69.3 | 69.2 |
| MLP balanced | 74.5 | 71.8 | 68.4 | 68.8 |
| TF-IDF | 81.6 | 80.0 | 77.5 | 77.4 |
| Emb.+TF-IDF | 81.7 | 80.1 | 77.6 | 77.4 |

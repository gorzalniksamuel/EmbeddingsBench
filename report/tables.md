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
| HUPD patents (IPC subclass) | EN | patents (IPC subclasses, long tail) | 408 | 98,195 / 18,934 / 18,901 | 151 | 50:1 |

## Classifiers against LogReg

| Classifier | Mean Δ (pts) | Median Δ (pts) | Better by > 0.5 pts | Datasets |
|---|---|---|---|---|
| FT+TF-IDF | +4.8 | +5.7 | 5 | 6 |
| Emb.+TF-IDF | +3.5 | +3.0 | 13 | 18 |
| Fine-tuned | +3.5 | +4.0 | 5 | 6 |
| TF-IDF | +0.5 | +0.3 | 9 | 18 |
| MLP (tuned) | +0.3 | +0.0 | 5 | 18 |
| RBF SVM | -0.9 | -0.1 | 3 | 18 |
| kNN | -6.5 | -5.3 | 0 | 18 |
| Zero-shot | -39.9 | -40.9 | 0 | 18 |

## Fine-tuning

| Dataset | Frozen LogReg | Fine-tuned | Best without GPU training | FT + TF-IDF | gbert-large | Train minutes |
|---|---|---|---|---|---|---|
| Amazon reviews DE | 62.2 | 63.6 | 63.3 | 63.7 | 64.7 | 37 |
| 10kGNAD German news | 84.1 | 88.6 | 88.7 | 89.9 | 91.5 | 76 |
| MASSIVE intents DE | 87.0 | 87.0 | 87.4 | 87.3 | 88.0 | 26 |
| Yahoo! Answers Topics | 67.1 | 71.5 | 70.8 | 72.9 | – | 39 |
| arXiv papers | 82.9 | 86.6 | 87.7 | 88.6 | – | 150 |
| 20 Newsgroups | 65.7 | 72.7 | 75.0 | 75.6 | – | 46 |

## Classifiers on frozen vs fine-tuned jina embeddings (test accuracy, %)

| Dataset | Frozen kNN | Frozen LogReg | Frozen RBF SVM | Frozen MLP (tuned) | FT-emb kNN | FT-emb LogReg | FT-emb RBF SVM | FT-emb MLP (tuned) | Fine-tuned | gbert-large |
|---|---|---|---|---|---|---|---|---|---|---|
| Amazon reviews DE | 62.1 | 62.2 | 62.2 | 62.2 | 60.3 | 63.3 | 62.7 | 63.5 | 63.6 | 64.7 |
| 10kGNAD German news | 74.3 | 84.1 | 82.4 | 83.3 | 87.0 | 87.9 | 88.7 | 87.8 | 88.6 | 91.5 |
| MASSIVE intents DE | 87.2 | 87.0 | 86.9 | 87.1 | 87.1 | 87.0 | 86.9 | 87.1 | 87.0 | 88.0 |
| Yahoo! Answers Topics | 49.2 | 67.1 | 65.2 | 67.0 | 65.1 | 71.1 | 71.4 | 71.2 | 71.5 | – |
| arXiv papers | 75.7 | 82.9 | 82.5 | 83.7 | 85.5 | 85.6 | 85.9 | 86.0 | 86.6 | – |
| 20 Newsgroups | 46.2 | 65.7 | 64.0 | 65.5 | 71.8 | 71.8 | 71.9 | 72.1 | 72.7 | – |

## Class imbalance (accuracy / balanced accuracy / macro-F1 / tail recall, %)

**Swiss Federal Supreme Court decisions DE (legal area)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 89.0 | 83.9 | 81.4 | 79.4 |
| LogReg weighted | 87.5 | 86.1 | 79.6 | 87.7 |
| Logit-adjusted | 89.0 | 83.9 | 81.4 | 79.4 |
| MLP (tuned) | 89.5 | 83.6 | 80.5 | 80.4 |
| MLP weighted | 86.7 | 84.4 | 78.1 | 88.2 |
| MLP balanced | 87.8 | 85.1 | 80.3 | 84.9 |
| TF-IDF | 92.2 | 86.6 | 86.0 | 83.9 |
| Emb.+TF-IDF | 93.3 | 88.0 | 87.5 | 85.3 |

**MTOP intents DE**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 94.5 | 81.8 | 80.6 | 27.9 |
| LogReg weighted | 94.5 | 86.0 | 82.0 | 36.8 |
| Logit-adjusted | 93.7 | 87.0 | 78.4 | 40.2 |
| MLP (tuned) | 94.4 | 83.9 | 81.9 | 33.8 |
| MLP weighted | 93.8 | 85.5 | 79.0 | 36.8 |
| MLP balanced | 94.4 | 86.4 | 83.9 | 37.7 |
| TF-IDF | 90.7 | 69.8 | 67.1 | 16.2 |
| Emb.+TF-IDF | 94.4 | 81.7 | 80.5 | 27.9 |

**Open Food Facts DE (product categories)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 44.9 | 42.4 | 42.8 | 36.4 |
| LogReg weighted | 42.5 | 46.2 | 42.3 | 46.8 |
| Logit-adjusted | 44.9 | 42.4 | 42.8 | 36.4 |
| MLP (tuned) | 46.4 | 43.4 | 43.7 | 38.3 |
| MLP weighted | 41.6 | 47.5 | 41.9 | 51.4 |
| MLP balanced | 42.3 | 47.7 | 43.0 | 50.7 |
| TF-IDF | 57.8 | 56.5 | 55.1 | 53.4 |
| Emb.+TF-IDF | 58.6 | 56.9 | 55.7 | 53.9 |

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

**HUPD patents (IPC subclass)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 50.2 | 33.1 | 32.9 | 12.6 |
| LogReg weighted | 44.3 | 40.5 | 33.7 | 29.8 |
| Logit-adjusted | 48.8 | 38.4 | 34.3 | 22.2 |
| MLP (tuned) | 50.8 | 38.7 | 36.5 | 24.6 |
| MLP weighted | 40.7 | 42.8 | 31.5 | 39.3 |
| MLP balanced | 42.7 | 42.9 | 33.7 | 34.9 |
| TF-IDF | 52.3 | 39.6 | 38.3 | 25.3 |
| Emb.+TF-IDF | 54.3 | 39.2 | 38.7 | 22.7 |

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
| FT+TF-IDF | +3.9 | +3.3 | 7 | 9 |
| Emb.+TF-IDF | +3.5 | +3.0 | 13 | 18 |
| Fine-tuned | +2.7 | +3.1 | 7 | 10 |
| TF-IDF | +0.5 | +0.3 | 9 | 18 |
| MLP (tuned) | +0.3 | +0.0 | 5 | 18 |
| RBF SVM | -0.9 | -0.1 | 3 | 18 |
| kNN | -6.5 | -5.3 | 0 | 18 |
| Zero-shot | -39.9 | -40.9 | 0 | 18 |

## Fine-tuning

| Dataset | Frozen LogReg | Fine-tuned | Best without GPU training | FT + TF-IDF | gbert-large | Train minutes |
|---|---|---|---|---|---|---|
| GermEval 2018 offensive language | 70.2 | 73.9 | 71.0 | 72.9 | – | 10 |
| Amazon reviews DE | 62.2 | 63.6 | 63.3 | 63.7 | 64.7 | 37 |
| 10kGNAD German news | 84.1 | 88.6 | 88.7 | 89.9 | 91.5 | 76 |
| MASSIVE intents DE | 87.0 | 87.0 | 87.4 | 87.3 | 88.0 | 26 |
| MTOP intents DE | 94.5 | 94.3 | 94.8 | – | – | 25 |
| Yahoo! Answers Topics | 67.1 | 71.5 | 70.8 | 72.9 | – | 39 |
| arXiv papers | 82.9 | 86.6 | 87.7 | 88.6 | – | 150 |
| 20 Newsgroups | 65.7 | 72.7 | 75.0 | 75.6 | – | 46 |
| Banking77 | 93.0 | 92.6 | 93.4 | 92.7 | – | 19 |
| CLINC150 | 92.6 | 95.2 | 94.8 | 95.9 | – | 34 |

## Classifiers on frozen vs fine-tuned jina embeddings (test accuracy, %)

| Dataset | Frozen kNN | Frozen LogReg | Frozen RBF SVM | Frozen MLP (tuned) | FT-emb kNN | FT-emb LogReg | FT-emb RBF SVM | FT-emb MLP (tuned) | Fine-tuned | gbert-large |
|---|---|---|---|---|---|---|---|---|---|---|
| GermEval 2018 offensive language | 69.0 | 70.2 | 69.4 | 70.1 | 71.6 | 73.0 | 73.2 | 72.1 | 73.9 | – |
| Amazon reviews DE | 62.1 | 62.2 | 62.2 | 62.2 | 60.3 | 63.3 | 62.7 | 63.5 | 63.6 | 64.7 |
| 10kGNAD German news | 74.3 | 84.1 | 82.4 | 83.3 | 87.0 | 87.9 | 88.7 | 87.8 | 88.6 | 91.5 |
| MASSIVE intents DE | 87.2 | 87.0 | 86.9 | 87.1 | 87.1 | 87.0 | 86.9 | 87.1 | 87.0 | 88.0 |
| Yahoo! Answers Topics | 49.2 | 67.1 | 65.2 | 67.0 | 65.1 | 71.1 | 71.4 | 71.2 | 71.5 | – |
| arXiv papers | 75.7 | 82.9 | 82.5 | 83.7 | 85.5 | 85.6 | 85.9 | 86.0 | 86.6 | – |
| 20 Newsgroups | 46.2 | 65.7 | 64.0 | 65.5 | 71.8 | 71.8 | 71.9 | 72.1 | 72.7 | – |
| Banking77 | 92.7 | 93.0 | 92.8 | 92.8 | 92.8 | 92.8 | 92.8 | 92.6 | 92.6 | – |
| CLINC150 | 87.4 | 92.6 | 92.9 | 92.9 | 92.8 | 94.7 | 94.8 | 95.0 | 95.2 | – |

## Significance: best method per dataset with 95 % bootstrap interval

| Dataset | Best (95 % interval) | Half-width (pts) | Statistically tied with the best |
|---|---|---|---|
| GermEval 2018 offensive language | Fine-tuned 73.9 [72.5–75.4] | ±1.4 | – |
| Amazon reviews DE | gbert-large 64.7 [63.2–66.0] | ±1.4 | Fine-tuned, Emb.+char TF-IDF, FT+TF-IDF |
| 10kGNAD German news | gbert-large 91.5 [89.9–93.2] | ±1.7 | – |
| Swiss Federal Supreme Court decisions DE | Emb.+char TF-IDF 93.7 [92.9–94.6] | ±0.8 | Emb.+TF-IDF, TF-IDF char |
| MASSIVE intents DE | gbert-large 88.0 [86.9–89.2] | ±1.2 | kNN, Emb.+TF-IDF, MLP (tuned), Emb.+char TF-IDF, FT+TF-IDF |
| MTOP intents DE | RBF SVM 94.8 [93.8–95.8] | ±1.0 | kNN, LogReg, MLP, Emb.+TF-IDF, Emb.+char TF-IDF, MLP (tuned), LogReg weighted, MLP balanced, Fine-tuned, FT weighted |
| Open Food Facts DE | Emb.+char TF-IDF 60.6 [59.7–61.7] | ±1.0 | TF-IDF char |
| AG News | Emb.+char TF-IDF 90.8 [90.1–91.4] | ±0.6 | Emb.+TF-IDF |
| Medical abstracts | TF-IDF char 77.5 [75.6–79.4] | ±1.9 | Emb.+TF-IDF, Emb.+char TF-IDF |
| Emotion | Emb.+char TF-IDF 92.7 [91.5–93.9] | ±1.2 | kNN, LogReg, MLP, RBF SVM, Emb.+TF-IDF, LogReg weighted, Logit-adjusted, MLP weighted, MLP balanced |
| Yahoo! Answers Topics | FT+TF-IDF 72.9 [71.7–74.0] | ±1.2 | – |
| arXiv papers | FT+TF-IDF 88.6 [87.4–89.9] | ±1.3 | – |
| DBpedia-14 | Emb.+TF-IDF 98.7 [98.2–99.1] | ±0.4 | – |
| 20 Newsgroups | FT+TF-IDF 75.6 [74.6–76.6] | ±1.0 | Emb.+char TF-IDF |
| Banking77 | Emb.+TF-IDF 93.4 [92.5–94.3] | ±0.9 | Emb.+char TF-IDF, MLP balanced |
| LEDGAR contract clauses | Emb.+TF-IDF 81.7 [81.0–82.5] | ±0.7 | TF-IDF |
| CLINC150 | FT+TF-IDF 95.9 [95.3–96.4] | ±0.6 | – |
| HUPD patents | Emb.+TF-IDF 54.3 [53.5–55.0] | ±0.7 | – |

## Calibration and selective prediction (%)

| Dataset | Classifier | Accuracy | ECE | Accuracy on the 80 % most confident | Automatable at 95 % accuracy | at 99 % |
|---|---|---|---|---|---|---|
| GermEval 2018 offensive language | LogReg | 70.2 | 3.1 | 76.9 | 7 | 0 |
| GermEval 2018 offensive language | Fine-tuned | 73.9 | 16.2 | 80.3 | 6 | 0 |
| Amazon reviews DE | LogReg | 62.2 | 4.4 | 66.6 | 1 | 0 |
| Amazon reviews DE | Fine-tuned | 63.6 | 3.5 | 68.0 | 4 | 0 |
| 10kGNAD German news | LogReg | 84.1 | 6.6 | 89.8 | 55 | 26 |
| 10kGNAD German news | Fine-tuned | 88.6 | 6.5 | 95.3 | 84 | 41 |
| Swiss Federal Supreme Court decisions DE | LogReg | 89.0 | 3.0 | 97.0 | 86 | 64 |
| MASSIVE intents DE | LogReg | 87.0 | 3.6 | 94.0 | 43 | 0 |
| MASSIVE intents DE | Fine-tuned | 87.0 | 3.6 | 94.0 | 40 | 0 |
| MTOP intents DE | LogReg | 94.5 | 2.1 | 98.7 | 99 | 66 |
| MTOP intents DE | Fine-tuned | 94.3 | 3.9 | 98.8 | 99 | 64 |
| Open Food Facts DE | LogReg | 44.9 | 3.8 | 52.7 | 2 | 0 |
| AG News | LogReg | 87.2 | 2.6 | 93.7 | 71 | 7 |
| Medical abstracts | LogReg | 70.5 | 6.0 | 76.6 | 12 | 0 |
| Emotion | LogReg | 92.4 | 9.0 | 98.5 | 93 | 71 |
| Yahoo! Answers Topics | LogReg | 67.1 | 5.7 | 75.2 | 11 | 1 |
| Yahoo! Answers Topics | Fine-tuned | 71.5 | 5.8 | 80.0 | 21 | 1 |
| arXiv papers | LogReg | 82.9 | 6.3 | 90.3 | 61 | 16 |
| arXiv papers | Fine-tuned | 86.6 | 5.8 | 93.7 | 76 | 51 |
| DBpedia-14 | LogReg | 98.2 | 1.1 | 99.6 | 100 | 99 |
| 20 Newsgroups | LogReg | 65.7 | 9.7 | 75.3 | 23 | 2 |
| 20 Newsgroups | Fine-tuned | 72.7 | 13.0 | 83.0 | 44 | 14 |
| Banking77 | LogReg | 93.0 | 1.3 | 98.8 | 96 | 72 |
| Banking77 | Fine-tuned | 92.6 | 4.8 | 98.6 | 95 | 73 |
| LEDGAR contract clauses | LogReg | 74.2 | 15.8 | 82.2 | 41 | 12 |
| CLINC150 | LogReg | 92.6 | 6.0 | 98.6 | 95 | 76 |
| CLINC150 | Fine-tuned | 95.2 | 2.1 | 99.4 | 100 | 86 |
| HUPD patents | LogReg | 50.2 | 8.4 | 56.7 | 0 | 0 |

## Class imbalance (accuracy / balanced accuracy / macro-F1 / tail recall, %)

**GermEval 2018 offensive language**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 70.2 | 35.5 | 37.2 | 2.2 |
| LogReg weighted | 62.3 | 45.2 | 41.8 | 17.8 |
| Logit-adjusted | 69.5 | 39.2 | 41.8 | 8.9 |
| MLP (tuned) | 70.1 | 36.7 | 38.9 | 4.4 |
| MLP weighted | 53.5 | 41.8 | 35.9 | 26.7 |
| MLP balanced | 62.2 | 43.7 | 41.2 | 13.3 |
| Fine-tuned | 73.9 | 44.1 | 47.4 | 11.1 |
| TF-IDF | 67.5 | 32.9 | 34.4 | 4.4 |
| Emb.+TF-IDF | 71.0 | 33.9 | 34.9 | 0.0 |

**Amazon reviews DE (stars)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 62.2 | 62.2 | 61.8 | 74.8 |
| LogReg weighted | 62.2 | 62.2 | 61.8 | 74.8 |
| Logit-adjusted | 62.2 | 62.2 | 61.8 | 74.8 |
| MLP (tuned) | 62.2 | 62.2 | 61.8 | 74.3 |
| MLP weighted | 62.2 | 62.2 | 61.8 | 74.3 |
| MLP balanced | 61.7 | 61.7 | 61.3 | 75.8 |
| Fine-tuned | 63.6 | 63.6 | 63.1 | 73.3 |
| TF-IDF | 54.6 | 54.6 | 53.6 | 70.7 |
| Emb.+TF-IDF | 63.0 | 63.0 | 62.5 | 75.8 |

**10kGNAD German news**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 84.1 | 82.5 | 82.8 | 73.9 |
| LogReg weighted | 82.6 | 82.3 | 81.5 | 79.3 |
| Logit-adjusted | 83.7 | 82.9 | 82.4 | 77.4 |
| MLP (tuned) | 83.3 | 81.9 | 82.4 | 75.0 |
| MLP weighted | 82.1 | 82.4 | 81.1 | 80.8 |
| MLP balanced | 82.5 | 82.4 | 81.5 | 79.6 |
| Fine-tuned | 88.6 | 88.1 | 88.0 | 85.8 |
| TF-IDF | 87.4 | 87.4 | 87.9 | 86.1 |
| Emb.+TF-IDF | 88.7 | 88.6 | 89.0 | 87.2 |

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

**MASSIVE intents DE**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 87.0 | 84.0 | 82.4 | 72.0 |
| LogReg weighted | 86.7 | 84.7 | 81.9 | 74.7 |
| Logit-adjusted | 87.0 | 84.0 | 82.4 | 72.0 |
| MLP (tuned) | 87.1 | 84.0 | 83.8 | 72.0 |
| MLP weighted | 86.7 | 84.7 | 81.8 | 74.7 |
| MLP balanced | 86.9 | 84.9 | 82.1 | 74.8 |
| Fine-tuned | 87.0 | 83.9 | 82.4 | 71.8 |
| TF-IDF | 78.6 | 72.6 | 73.9 | 60.7 |
| Emb.+TF-IDF | 87.3 | 83.6 | 82.3 | 70.5 |

**MTOP intents DE**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 94.5 | 81.8 | 80.6 | 27.9 |
| LogReg weighted | 94.5 | 86.0 | 82.0 | 36.8 |
| Logit-adjusted | 93.7 | 87.0 | 78.4 | 40.2 |
| MLP (tuned) | 94.4 | 83.9 | 81.9 | 33.8 |
| MLP weighted | 93.8 | 85.5 | 79.0 | 36.8 |
| MLP balanced | 94.4 | 86.4 | 83.9 | 37.7 |
| Fine-tuned | 94.3 | 81.8 | 79.8 | 26.5 |
| FT weighted | 94.7 | 84.8 | 82.3 | 30.9 |
| FT logit-adj. | 94.8 | 85.7 | 81.9 | 33.8 |
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

**AG News**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 87.2 | 87.2 | 87.2 | 82.1 |
| LogReg weighted | 87.2 | 87.2 | 87.2 | 82.1 |
| Logit-adjusted | 87.2 | 87.2 | 87.2 | 82.1 |
| MLP (tuned) | 87.3 | 87.3 | 87.3 | 82.1 |
| MLP weighted | 87.3 | 87.3 | 87.3 | 82.1 |
| MLP balanced | 86.9 | 86.9 | 86.9 | 79.9 |
| TF-IDF | 89.0 | 89.0 | 89.0 | 83.9 |
| Emb.+TF-IDF | 90.4 | 90.4 | 90.4 | 85.6 |

**Medical abstracts**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 70.5 | 67.3 | 69.1 | 53.0 |
| LogReg weighted | 70.5 | 73.6 | 69.6 | 81.9 |
| Logit-adjusted | 71.5 | 71.6 | 70.6 | 68.1 |
| MLP (tuned) | 71.6 | 70.6 | 70.5 | 68.1 |
| MLP weighted | 70.3 | 74.1 | 68.9 | 86.1 |
| MLP balanced | 70.4 | 72.4 | 69.4 | 78.9 |
| TF-IDF | 75.6 | 73.2 | 74.8 | 66.9 |
| Emb.+TF-IDF | 76.4 | 73.8 | 75.5 | 67.5 |

**Emotion (tweets)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 92.4 | 86.2 | 88.2 | 70.1 |
| LogReg weighted | 92.3 | 90.0 | 88.5 | 85.8 |
| Logit-adjusted | 92.5 | 90.3 | 88.9 | 85.6 |
| MLP (tuned) | 91.7 | 87.2 | 87.6 | 76.0 |
| MLP weighted | 92.4 | 90.9 | 88.9 | 89.1 |
| MLP balanced | 92.3 | 90.5 | 88.7 | 87.1 |
| TF-IDF | 88.6 | 81.8 | 82.9 | 67.6 |
| Emb.+TF-IDF | 92.2 | 85.8 | 87.4 | 71.1 |

**Yahoo! Answers Topics**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 67.1 | 67.1 | 66.9 | 59.1 |
| LogReg weighted | 67.1 | 67.1 | 66.9 | 59.1 |
| Logit-adjusted | 67.1 | 67.1 | 66.9 | 59.1 |
| MLP (tuned) | 67.0 | 67.0 | 66.3 | 57.3 |
| MLP weighted | 67.0 | 67.0 | 66.3 | 57.3 |
| MLP balanced | 66.9 | 66.9 | 66.1 | 54.0 |
| Fine-tuned | 71.5 | 71.5 | 71.2 | 62.7 |
| TF-IDF | 64.5 | 64.5 | 63.9 | 55.8 |
| Emb.+TF-IDF | 70.0 | 70.0 | 69.6 | 60.3 |

**arXiv papers (11 categories)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 82.9 | 82.3 | 82.2 | 72.8 |
| LogReg weighted | 82.9 | 82.3 | 82.2 | 72.8 |
| Logit-adjusted | 82.9 | 82.3 | 82.2 | 72.8 |
| MLP (tuned) | 83.7 | 83.2 | 83.0 | 74.0 |
| MLP weighted | 83.7 | 83.2 | 83.0 | 74.0 |
| MLP balanced | 83.4 | 82.8 | 82.6 | 72.0 |
| Fine-tuned | 86.6 | 86.1 | 85.9 | 79.9 |
| TF-IDF | 87.3 | 86.7 | 86.5 | 77.2 |
| Emb.+TF-IDF | 87.7 | 87.1 | 87.0 | 78.2 |

**DBpedia-14**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 98.2 | 98.2 | 98.2 | 98.5 |
| LogReg weighted | 98.2 | 98.2 | 98.2 | 98.5 |
| Logit-adjusted | 98.2 | 98.2 | 98.2 | 98.5 |
| MLP (tuned) | 98.1 | 98.1 | 98.1 | 98.2 |
| MLP weighted | 98.1 | 98.1 | 98.1 | 98.2 |
| MLP balanced | 98.3 | 98.3 | 98.3 | 98.2 |
| TF-IDF | 96.8 | 96.7 | 96.7 | 96.2 |
| Emb.+TF-IDF | 98.7 | 98.7 | 98.7 | 98.8 |

**20 Newsgroups**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 65.7 | 64.3 | 64.1 | 49.2 |
| LogReg weighted | 65.5 | 64.4 | 64.5 | 51.3 |
| Logit-adjusted | 65.2 | 64.1 | 64.3 | 51.5 |
| MLP (tuned) | 65.5 | 64.2 | 64.1 | 49.6 |
| MLP weighted | 65.4 | 64.3 | 64.4 | 50.8 |
| MLP balanced | 65.6 | 64.4 | 64.5 | 50.9 |
| Fine-tuned | 72.7 | 71.3 | 71.2 | 55.3 |
| TF-IDF | 71.6 | 70.2 | 70.0 | 57.1 |
| Emb.+TF-IDF | 73.6 | 72.2 | 71.9 | 57.8 |

**Banking77**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 93.0 | 93.0 | 93.0 | 94.0 |
| LogReg weighted | 93.0 | 93.0 | 92.9 | 94.0 |
| Logit-adjusted | 93.0 | 93.0 | 93.0 | 94.0 |
| MLP (tuned) | 92.8 | 92.8 | 92.7 | 93.7 |
| MLP weighted | 92.9 | 92.9 | 92.8 | 94.2 |
| MLP balanced | 93.0 | 93.0 | 92.9 | 94.4 |
| Fine-tuned | 92.6 | 92.6 | 92.6 | 93.4 |
| TF-IDF | 88.5 | 88.5 | 88.4 | 88.3 |
| Emb.+TF-IDF | 93.4 | 93.3 | 93.3 | 94.4 |

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

**CLINC150 (in-scope)**


| Classifier | Accuracy | Balanced acc. | Macro-F1 | Tail recall |
|---|---|---|---|---|
| LogReg | 92.6 | 92.6 | 92.6 | 93.3 |
| LogReg weighted | 92.6 | 92.6 | 92.6 | 93.3 |
| Logit-adjusted | 92.6 | 92.6 | 92.6 | 93.3 |
| MLP (tuned) | 92.9 | 92.9 | 92.9 | 93.6 |
| MLP weighted | 92.9 | 92.9 | 92.9 | 93.6 |
| MLP balanced | 92.7 | 92.7 | 92.7 | 93.3 |
| Fine-tuned | 95.2 | 95.2 | 95.2 | 95.5 |
| TF-IDF | 91.3 | 91.3 | 91.2 | 93.2 |
| Emb.+TF-IDF | 94.8 | 94.8 | 94.8 | 95.3 |

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

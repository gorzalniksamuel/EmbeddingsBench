#!/usr/bin/env bash
# Full benchmark from scratch: data → embeddings → frozen-embedding classifiers (incl. character TF-IDF, MLP search,
# imbalance schemes) → optional fine-tuning and gbert → fine-tuned ensembles → metrics, model registration, costs → publish.
set -e
cd "$(dirname "$0")"
set -a; [ -f ../.env ] && . ../.env; set +a
PY="${EB_PYTHON:-python}"
"$PY" prepare_raw.py                      # datasets that need cleaning (HUPD, Open Food Facts, Swiss case law)
"$PY" data.py
"$PY" embed.py
"$PY" evaluate.py
"$PY" extras.py
"$PY" tfidf_char.py
"$PY" mlp_search.py --trials "${EB_MLP_TRIALS:-40}"
"$PY" imbalance.py ${EB_IMBALANCED:-ledgar mtop_intent_de swiss_caselaw_de off_de hupd_ipc}
for ds in ${EB_FINETUNE:-}; do
  "$PY" finetune.py "$ds" --epochs "${EB_FT_EPOCHS:-3}"
done
for ds in ${EB_GBERT:-}; do
  "$PY" gbert.py "$ds"
done
if [ -n "${EB_FINETUNE:-}" ]; then "$PY" ft_ensemble.py; fi
"$PY" backfill_metrics.py
"$PY" costs.py
./publish.sh
echo "benchmark complete"

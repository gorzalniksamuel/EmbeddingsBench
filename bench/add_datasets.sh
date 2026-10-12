#!/usr/bin/env bash
# Build, embed and benchmark newly defined datasets (incl. imbalance schemes), then publish.
# usage: add_datasets.sh <dataset> [...]   (datasets defined in data.py SPECS but not yet in results.json)
set -e
cd "$(dirname "$0")"
set -a; [ -f ../.env ] && . ../.env; set +a
PY="${EB_PYTHON:-python}"
"$PY" data.py
"$PY" embed.py
"$PY" evaluate.py "$@"
"$PY" extras.py "$@"
"$PY" tfidf_char.py "$@"
"$PY" mlp_search.py --trials "${EB_MLP_TRIALS:-40}" "$@"
"$PY" imbalance.py "$@"
"$PY" backfill_metrics.py
./publish.sh
echo "new datasets done: $*"

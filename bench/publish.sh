#!/usr/bin/env bash
# Copy the current results and prediction files to the web root (atomic per file).
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
SITE="${EB_SITE_DIR:-site}"
mkdir -p "$SITE/predictions"
cp results/results.json $SITE/results.json.tmp && mv $SITE/results.json.tmp $SITE/results.json
for f in results/predictions/*.json.gz; do cp "$f" "$SITE/predictions/$(basename "$f").tmp" && mv "$SITE/predictions/$(basename "$f").tmp" "$SITE/predictions/$(basename "$f")"; done

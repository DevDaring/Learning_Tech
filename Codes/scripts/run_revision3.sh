#!/usr/bin/env bash
# E10 -> E11 -> E12 tutor timing, one after another (docs/revision_plan_frozen.md, third addendum)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for s in e10 e11 e12api; do
  echo "===== $s start $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')"
  ~/venv/bin/python scripts/revision3_experiments.py $s || { echo "STAGE FAILED: $s"; exit 1; }
  echo "===== $s done $(TZ=Asia/Kolkata date '+%H:%M IST')"
done

#!/bin/bash
# Finish the full study: B5 (B1-B3 already written), human export, analysis, checks.
cd "$(dirname "$0")/.."
export PYTHONPATH=src TLT_R_CORES=8
~/venv/bin/python -W ignore -m tlt baselines --only B5 --allow-unreviewed || { echo "STAGE FAILED: baselines B5"; exit 1; }
for s in export_human analysis; do
  ~/venv/bin/python -W ignore -m tlt $s --allow-unreviewed || { echo "STAGE FAILED: $s"; exit 1; }
done
~/venv/bin/python -W ignore -m tlt checks --allow-unreviewed; echo "checks exit=$?"
echo "FULL STUDY COMPLETE"

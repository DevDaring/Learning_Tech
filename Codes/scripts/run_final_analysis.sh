#!/bin/bash
# Final analysis after the human review was applied (all inputs already on disk).
cd "$(dirname "$0")/.."
export PYTHONPATH=src TLT_R_CORES=8
unset TLT_TRANSLATE_URL
~/venv/bin/python -W ignore -m tlt analysis --allow-unreviewed || { echo "STAGE FAILED: analysis"; exit 1; }
~/venv/bin/python -W ignore scripts/paper_extra_analysis.py > paper_extra.log 2>&1 || { echo "STAGE FAILED: extra analysis"; exit 1; }
~/venv/bin/python scripts/paper_figures.py Submission1/figures > paper_figures.log 2>&1 || { echo "STAGE FAILED: figures"; exit 1; }
~/venv/bin/python -W ignore -m tlt checks --allow-unreviewed; echo "checks exit=$?"
echo "FINAL ANALYSIS COMPLETE"

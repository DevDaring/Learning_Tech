#!/bin/bash
# Finish applying the human review: GPU stages, destroy our Vast instance, then the rest.
cd "$(dirname "$0")/.."
export PYTHONPATH=src TLT_R_CORES=8 TLT_TRANSLATE_URL=http://127.0.0.1:18000
run() { ~/venv/bin/python -W ignore -m tlt "$@" --allow-unreviewed || { echo "STAGE FAILED: $*"; exit 1; }; }
run translated_judging
run repair
echo "GPU PHASE COMPLETE"
~/venv/bin/python scripts/vast_translator.py down
unset TLT_TRANSLATE_URL
run auto_metrics
run baselines --only B1 B2 B5
while pgrep -f "tlt baselines --only B3" >/dev/null; do sleep 30; done
grep -q "B3 exit=0" b3_corrections.log || { echo "STAGE FAILED: baselines B3"; exit 1; }
run analysis
~/venv/bin/python -W ignore scripts/paper_extra_analysis.py > paper_extra.log 2>&1 || { echo "STAGE FAILED: extra analysis"; exit 1; }
~/venv/bin/python scripts/paper_figures.py Submission1/figures > /dev/null 2>&1 || { echo "STAGE FAILED: figures"; exit 1; }
~/venv/bin/python -W ignore -m tlt checks --allow-unreviewed; echo "checks exit=$?"
echo "CORRECTIONS COMPLETE"

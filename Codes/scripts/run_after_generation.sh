#!/bin/bash
# Full study after generation. GPU stages first (repair needs IndicTrans2 for R4/R5; translated
# judging needs the main judgements), then the Vast instance is destroyed, then CPU-only stages.
cd "$(dirname "$0")/.."
export TLT_TRANSLATE_URL=http://127.0.0.1:18000 PYTHONPATH=src
run() { ~/venv/bin/python -W ignore -m tlt "$1" --allow-unreviewed || { echo "STAGE FAILED: $1"; exit 1; }; }
for s in repair mask_and_judge translated_judging; do run $s; done
echo "GPU PHASE COMPLETE"
~/venv/bin/python scripts/vast_translator.py down
unset TLT_TRANSLATE_URL
for s in judge_validation auto_metrics baselines export_human analysis; do run $s; done
~/venv/bin/python -W ignore -m tlt checks --allow-unreviewed; echo "checks exit=$?"
echo "FULL STUDY COMPLETE"

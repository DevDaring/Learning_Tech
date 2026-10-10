#!/bin/bash
# Apply the returned human review: regenerate and re-judge only prompts whose text changed,
# then every dependent stage. No GPU: the few new IndicTrans2 segments run on the CPU.
cd "$(dirname "$0")/.."
export PYTHONPATH=src TLT_R_CORES=8
unset TLT_TRANSLATE_URL
run() { ~/venv/bin/python -W ignore -m tlt "$@" --allow-unreviewed || { echo "STAGE FAILED: $*"; exit 1; }; }
run generate
# B3 needs only the replies; run it alongside judging (it embeds on the CPU for ~3 h)
( ~/venv/bin/python -W ignore -m tlt baselines --only B3 --allow-unreviewed > b3_corrections.log 2>&1; echo "B3 exit=$?" >> b3_corrections.log ) &
B3=$!
run mask_and_judge
run translated_judging
run repair
run auto_metrics
run baselines --only B1 B2 B5
wait $B3; grep -q "B3 exit=0" b3_corrections.log || { echo "STAGE FAILED: baselines B3"; exit 1; }
run analysis
~/venv/bin/python -W ignore scripts/paper_extra_analysis.py > paper_extra.log 2>&1 || { echo "STAGE FAILED: extra analysis"; exit 1; }
~/venv/bin/python scripts/paper_figures.py Submission1/figures > /dev/null 2>&1 || { echo "STAGE FAILED: figures"; exit 1; }
~/venv/bin/python -W ignore -m tlt checks --allow-unreviewed; echo "checks exit=$?"
echo "CORRECTIONS COMPLETE"

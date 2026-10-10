#!/usr/bin/env bash
# E1 (docs/revision_plan_frozen.md): R2-L, R2-T and the R2/R5 drift reference, through the repair
# stage, using the Vast IndicTrans2 service opened by scripts/vast_translator.py up.
set -u
cd "$(dirname "$0")/.."
export TLT_TRANSLATE_URL="http://127.0.0.1:18000"
export TLT_TRANSLATE_TOKEN="$(~/venv/bin/python -c "import json;print(json.load(open('runs/vast_translator.json'))['token'])")"
PYTHONPATH=src ~/venv/bin/python -m tlt repair --allow-unreviewed
echo "exit=$?"

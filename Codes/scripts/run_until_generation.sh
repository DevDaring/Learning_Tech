#!/bin/bash
# Full study, stages up to and including generation (judging waits for the judge decision).
cd "$(dirname "$0")/.."
export TLT_TRANSLATE_URL=http://127.0.0.1:18000 PYTHONPATH=src
for s in translate build_prompts generate; do
  ~/venv/bin/python -W ignore -m tlt $s --allow-unreviewed || { echo "STAGE FAILED: $s"; exit 1; }
done
echo "GENERATION PHASE COMPLETE"

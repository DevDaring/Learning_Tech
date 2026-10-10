#!/usr/bin/env bash
# E4 now; E2 and E3 after the E1 repair stage has written its tables (they read them).
set -u
cd "$(dirname "$0")/.."
PYTHONPATH=src ~/venv/bin/python -u scripts/revision_experiments.py e4 > revision_e4.log 2>&1
echo "e4 exit=$?" >> revision_e4.log
until grep -q "^exit=" revision_e1.log; do sleep 30; done
PYTHONPATH=src ~/venv/bin/python scripts/revision_experiments.py e2 > revision_e2e3.log 2>&1
PYTHONPATH=src ~/venv/bin/python scripts/revision_experiments.py e3sem >> revision_e2e3.log 2>&1
PYTHONPATH=src ~/venv/bin/python scripts/revision_experiments.py e3abl >> revision_e2e3.log 2>&1
echo "e2e3 exit=$?" >> revision_e2e3.log

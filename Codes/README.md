# Same Question, Lesser Lesson — pipeline

Code for the study specified in `TLT_Expectation_Gap_Project.md`: measuring and repairing
pedagogical disparity in multilingual LLM tutors (target: IEEE TLT).

## Layout

```
config/    design.yaml (every count, seed and condition), models.yaml (models, routes, keys by NAME)
prompts/   every prompt as a text file, loaded by name (quote these in the appendix)
src/tlt/   the package: stages/ (one module per stage), analysis/, baselines/
data/      items, translations, prompts            runs/     append-only JSONL call logs
results/   tables/, figures/, baselines/, mixed/   review/   human review and annotation files
docs/      verification_log.md, deviations.md, name_validation.md, learner_study_protocol.md
tests/     unit tests (pytest) and a SYNTHETIC Eedi-shaped fixture used only by --dry-run
external/  cloned MRBench, MathDial and BEnQA repositories
```

## Setup

```bash
cd Codes
~/venv/bin/pip install -r requirements.txt
# R for the mixed models (user-space, no sudo); stage 9 finds it automatically:
#   ~/.local/micromamba/bin/micromamba create -r ~/.local/micromamba/root -n tlt_r -c conda-forge r-base=4.4 r-ordinal r-lme4 r-jsonlite
export PYTHONPATH=src
```

Keys are read from `Codes/.env` by variable name (`config/models.yaml` → `credentials`).
Logs record only the name of the credential used, and `checks` scans every output file to
confirm no key value was written.

### Two one-time actions before the pilot

1. **Eedi**: accept the rules at
   https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/rules, then
   `kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p data/raw/eedi && unzip -o data/raw/eedi/*.zip -d data/raw/eedi`
2. **IndicTrans2**: while logged in to the Hugging Face account whose token is in `.env`,
   accept the licence on https://huggingface.co/ai4bharat/indictrans2-en-indic-1B and
   https://huggingface.co/ai4bharat/indictrans2-indic-en-1B (approval is automatic).

## Running

```bash
python -m tlt <stage> [--pilot | --dry-run]
python -m tlt all --dry-run     # whole pipeline, mock models, no network, no cost (synthetic Eedi)
python -m tlt all --pilot       # 30 items, 1 sample, 2 producers, 3 judges, real calls
python -m tlt all               # full run (blocked until the human gates below are complete)
```

| # | Stage | Writes |
|---|---|---|
| 0 | `verify` | appends a dated section to `docs/verification_log.md`; `results/verification_probe.json` |
| 1 | `build_items` | `data/items.jsonl` (300 Eedi + 150 BEnQA, nested subsets A ⊃ B ⊃ C) |
| 2 | `translate` | `data/items_multilingual.jsonl`, `review/translation_review.csv` |
| 3 | `build_prompts` | `data/prompts.jsonl`, `review/template_approval.csv` |
| 4 | `generate` | `runs/generations.jsonl` |
| 5 | `mask_and_judge` | `runs/judgements.jsonl`, `results/judgements_{long,aggregated}.csv` |
| 6 | `judge_validation` | `results/judge_validation.csv` (B4, MRBench) |
| 7 | `translated_judging` | `results/translated_judging.csv` |
| 8 | `auto_metrics` | `results/auto_metrics.csv`, `results/repair_auto_metrics.csv` |
| 9 | `analysis` | `results/tables/*.csv`, `results/figures/*.{pdf,png}`, `results/mixed/` |
| 10 | `baselines` | `results/baselines/b1..b5*.csv` (`--only B1 B3` to run a subset) |
| 11 | `repair` | `runs/repair_generations.jsonl`, `results/repair_judgements_*.csv` |
| 12 | `export_human` | `review/human_annotation.csv` (+ separate key file) |
| — | `checks` | `results/acceptance.json` (Section 11 acceptance checks; non-zero exit on failure) |

`all` runs stages in dependency order: repair comes before auto_metrics and baselines,
because B5 and Table 9 need the repaired replies. Every stage resumes: completed calls are
keyed in the append-only logs and never repeated, and failed calls are retried on rerun (at
most 3 attempts in total).

## Review pack for human reviewers

```bash
python -m tlt review_pack      # -> review/outgoing/TLT_human_review_pack_<date>.zip
python -m tlt import_review --files Hindi_template_review_AB.xlsx Bengali_template_review_CD.xlsx Name_survey_EF.xlsx Name_survey_GH.xlsx
```

The zip holds `INSTRUCTIONS.md` (for the reviewers), Hindi and Bengali template-review workbooks
(one native speaker each) and a blind name survey (two independent annotators; study surnames
mixed with fillers, intended groups never shown). `import_review` writes approvals into
`review/template_approval.csv`; a row is approved only if the reviewer approved it, typed no
correction and, for Hindi student-facing rows, said it does not reveal gender. Name surveys go
to `review/name_survey/`, with an agreement summary in `review/name_survey_summary.csv`.
Record the outcome in `docs/name_validation.md` yourself. After editing any template, rebuild
the pack: rows whose text changed are imported as `stale` and need a fresh review.

## Human steps (Section 4.2, 4.3, 6.7, 7.3)

| File | Who | What |
|---|---|---|
| `review/template_approval.csv` | native speaker | set `status` to `approved` for every Hindi, Bengali and code-mixed template and name rendering |
| `review/translation_review.csv` | study owner | review every `flagged` and `review_sample` row; set `approved`, or `corrected` and fill `corrected_text`; then rerun `translate` and `build_prompts` |
| `docs/name_validation.md` | study owner + 2 annotators | resolve the four unsupported surnames, record confirmations, set `STATUS: CONFIRMED` |
| `review/human_annotation.csv` | 2-3 maths teachers | each returns a filled copy as `review/human_returned/<name>.csv`; until then every table is marked PROVISIONAL |

## Volume (from config/design.yaml)

| Block | Prompts | Generations |
|---|---|---|
| A | 1,500 | 27,000 |
| B (new) | 2,700 | 32,400 |
| C | 200 | 2,400 |
| D (incl. BN-MT, D2) | 450 | 5,400 |
| Repair (R1-R5 × 5 languages, + identity R5) | 2,700 cells | 16,200 |

Judge calls (GLM-4.7, DeepSeek-V3.2, Qwen3-Next 80B on AWS Bedrock; deviations D34): 3 per generation, plus 20% translated judging, plus 1,500 for MRBench.
`build_prompts` prints the exact counts. `docs/deviations.md` lists every departure from the
specification and its reason.

## Tests

```bash
~/venv/bin/python -m pytest -q tests
```

"""Stage 12: human-anchor export (Section 6.7).

review/human_annotation.csv: language (5) x producer (6) x 10 Block A replies, seeded,
shuffled, in the judges' blind format, rubric columns empty. The mapping from row_id to
generation is written separately to review/human_annotation_key.csv and must not be
shared with annotators. Each annotator returns a copy as
review/human_returned/<annotator>.csv (same columns, rubric filled in). The rubric
definitions given to the judges are copied to review/human_annotation_guidelines.txt.
"""
from __future__ import annotations

import logging
import random

import pandas as pd

from ..config import Run
from ..io import prompt, read_jsonl
from ..names import mask_names
from ..rubric import DIMENSIONS
from .s04_generate import current_generations

log = logging.getLogger("tlt.s12")


def run_stage(run: Run) -> dict:
    # Never re-draw the sample once teachers have returned ratings: a new sample would replace
    # the export and the private key that their ratings refer to.
    returned = sorted(run.path("review", "human_returned", "x").parent.glob("*.csv"))
    if returned and run.path("review", "human_annotation_key.csv").exists():
        log.info("human export kept: %d returned annotation files refer to the existing export", len(returned))
        return {"exported": 0, "kept_existing": True, "returned_files": len(returned)}
    agg_path = run.path("results", "judgements_aggregated.csv")
    gens = {g["call_id"]: g for g in current_generations(run, ok_only=True)}
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    per_cell = int(run.design["human_export"]["per_cell"])
    rng = random.Random(run.seed + 12)
    pool = [g for g in gens.values() if g["block"] == "A"]
    chosen = []
    for lang in run.design["languages"]["core"]:
        for m in run.producers:
            cell = sorted((g for g in pool if g["student_lang"] == lang and g["model"] == m),
                          key=lambda g: g["call_id"])
            # at most one reply per item within a cell, so the anchor spans many items
            seen, uniq = set(), []
            for g in rng.sample(cell, len(cell)):
                if g["item_id"] not in seen:
                    seen.add(g["item_id"])
                    uniq.append(g)
            chosen.extend(uniq[:per_cell])
    rng.shuffle(chosen)
    rows, key = [], []
    for i, g in enumerate(chosen, 1):
        p = prompts[g["prompt_id"]]
        rid = f"H{i:04d}"
        rows.append({"row_id": rid, "question_en": p["judge_question_en"], "correct_answer": p["judge_correct_answer"],
                     "student_answer": p["judge_student_answer"], "misconception": p["misconception_name"],
                     "student_wrote_in": p["language_label"], "tutor_reply": mask_names(g["raw_text"] or "")[0],
                     **{d: "" for d in DIMENSIONS}, "annotator": "", "notes": ""})
        key.append({"row_id": rid, "target_id": g["call_id"], "item_id": g["item_id"], "language": g["student_lang"],
                    "model": g["model"], "sample_index": g["sample_index"]})
    pd.DataFrame(rows).to_csv(run.path("review", "human_annotation.csv"), index=False)
    pd.DataFrame(key).to_csv(run.path("review", "human_annotation_key.csv"), index=False)
    allowed = pd.DataFrame([{"dimension": d, "allowed_labels": " | ".join(v)} for d, v in DIMENSIONS.items()])
    allowed.to_csv(run.path("review", "human_annotation_allowed_labels.csv"), index=False)
    run.path("review", "human_returned", "x").parent.mkdir(parents=True, exist_ok=True)
    run.path("review", "human_annotation_guidelines.txt").write_text(prompt("judge_definitions") + "\n",
                                                                     encoding="utf-8")
    stats = {"exported": len(rows), "judged_table_exists": agg_path.exists()}
    log.info("human export: %s", stats)
    return stats

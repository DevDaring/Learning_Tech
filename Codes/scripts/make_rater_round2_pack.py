"""Second, minimal round of blinded teacher ratings for the repair arms.

Part 1 re-uses the 24 R2-T/R5 pairs of the first round, so a second rater gives inter-rater
agreement on the paper's key human comparison. Part 2 adds 24 matched R2-T/R2-L pairs (same item,
tutor and language; 2 per tutor and language, Hindi and Bengali), drawn with a fixed seed before
any rating exists: R2-T replies are machine-translated back from English, R2-L replies are written
directly in the student's language, so Part 2 tests the naturalness of the recommended pivot design.
Pairs already in Part 1 are excluded from Part 2, so no reply is shown twice.

Rater A (a new rater who reads Hindi and Bengali) rates Parts 1 and 2 (48 pairs, mixed order).
Rater B (the first-round teacher, or another bilingual teacher) rates Part 2 only (24 pairs, own
order). Each reply gets the same six labels as the first round; each pair gets a blinded preference.
The packs hold no arm, tutor or study labels; the key is written separately and must not be sent.

Frozen analysis (fixed before any rating is returned):
  Part 1  agreement of rater A with the first-round teacher on pair preference (Cohen's kappa over
          A/B/No difference) and per-dimension exact agreement; pooled R2-T vs R5 preference over
          both raters (two-sided sign test on non-ties).
  Part 2  per rater and pooled: R2-T vs R2-L preference (two-sided sign test on non-ties); share of
          replies labelled Natural per arm and paired naturalness difference (sign test on the
          3-level ordinal label); human PQI-R per arm with a paired bootstrap interval; agreement of
          raters A and B (Cohen's kappa). With 24-48 pairs all results are reported as descriptive
          human evidence; no equivalence is claimed.

    PYTHONPATH=src python scripts/make_rater_round2_pack.py
Writes review/outgoing/round2/Rater_A_pack.xlsx, Rater_B_pack.xlsx and
review/repair_rating_key_round2.csv.
"""
import json
import random

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from gaptrace.config import load_run
from gaptrace.io import read_jsonl, read_table

run = load_run("full")
rng = random.Random(run.seed + 72)
rep = read_table(run.path("results", "repair_judgements_aggregated.csv"), low_memory=False)
rep = rep[(rep.variant == "original") & (rep.cue == "C0") & rep.condition.isin(["HI", "BN"])]

# Part 1: the first-round R2-T/R5 pairs, identified by their target ids.
k1 = pd.read_csv(run.path("review", "repair_rating_key.csv"))
part1 = []
for pair, g in k1.groupby("pair"):
    t = dict(zip(g.arm, g.target_id))
    r = g.iloc[0]
    part1.append({"part": 1, "item_id": r.item_id, "condition": r.language, "model": r.model,
                  "arms": ("R2T", "R5"), "targets": (t["R2T"], t["R5"]), "round1_pair": pair})
used = {(p["item_id"], p["condition"], p["model"]) for p in part1}

# Part 2: new R2-T/R2-L pairs, 2 per tutor and language.
r2 = rep[rep.arm.isin(["R2T", "R2L"])]
pairs = r2.pivot_table(index=["item_id", "condition", "model"], columns="arm", values="target_id",
                       aggfunc="first").dropna().reset_index()
pairs = pairs[[(a, b, c) not in used for a, b, c in zip(pairs.item_id, pairs.condition, pairs.model)]]
part2 = []
for (lang, model), g in pairs.sort_values(["condition", "model", "item_id"]).groupby(["condition", "model"]):
    for p in rng.sample(list(g.itertuples(index=False)), 2):
        part2.append({"part": 2, "item_id": p.item_id, "condition": lang, "model": model,
                      "arms": ("R2T", "R2L"), "targets": (p.R2T, p.R2L), "round1_pair": ""})

texts = {}
for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    texts[r["call_id"]] = r
items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}
prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}

LAB = {"Mistake_Identification": "Yes,To some extent,No", "Mistake_Location": "Yes,To some extent,No",
       "Revealing_of_the_Answer": "No,Yes (correct),Yes (incorrect)",
       "Providing_Guidance": "Yes,To some extent,No", "Math_Correctness": "Yes,No",
       "Language_Naturalness": "Natural,Understandable but unnatural,Hard to understand"}
INSTRUCTIONS = [
    "Rating tutor replies to a student's maths mistake (round 2)",
    "Read the enclosed Rater_Guidelines.pdf before you start; it defines every label with examples.",
    "Each row is one tutor reply to a student who sent the message shown, with a wrong answer.",
    "The correct answer is given for your reference; the tutor did not see it.",
    "Sheet 'Ratings': rate every reply on the six columns, using the drop-down lists only.",
    "Sheet 'Pairs': for each pair, say which reply (A or B) would help the student learn more,",
    "or 'No difference'. Rate the replies independently first, then the pairs.",
    "Work alone. Do not use AI tools, translators or search engines, and do not discuss ratings.",
    "Do not write your name anywhere in this file. Save it under the same file name and return it.",
]


def build(name, chosen, seed_offset):
    """Write one blinded workbook and return its key rows."""
    r = random.Random(run.seed + seed_offset)
    chosen = list(chosen)
    r.shuffle(chosen)
    rows, key = [], []
    for n, p in enumerate(chosen, start=1):
        order = list(zip(p["arms"], p["targets"]))
        r.shuffle(order)
        msg = prompts[f"A|{p['item_id']}|{p['condition']}|C0"]["user"]
        v = items[p["item_id"]]["versions"][p["condition"]]
        for label, (arm, tid) in zip("AB", order):
            rid = f"P{n:02d}{label}"
            rows.append({"row_id": rid, "pair": f"P{n:02d}", "reply": label,
                         "language": {"HI": "Hindi", "BN": "Bengali"}[p["condition"]],
                         "correct_answer": v["correct_answer"], "student_message": msg,
                         "tutor_reply": texts[tid].get("response_text") or ""})
            key.append({"workbook": name, "row_id": rid, "pair": f"P{n:02d}", "part": p["part"],
                        "round1_pair": p["round1_pair"], "arm": arm, "item_id": p["item_id"],
                        "language": p["condition"], "model": p["model"], "target_id": tid})
    assert all(x["tutor_reply"].strip() for x in rows), "empty reply in pack"

    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS, 1):
        ins.cell(row=i, column=1, value=line).font = Font(bold=(i == 1))
    ins.column_dimensions["A"].width = 100
    ws = wb.create_sheet("Ratings")
    cols = ["row_id", "pair", "reply", "language", "correct_answer", "student_message", "tutor_reply"] + list(LAB)
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for x in rows:
        ws.append([x[c] for c in cols[:7]] + [""] * len(LAB))
    for j, (dim, opts) in enumerate(LAB.items(), start=8):
        dv = DataValidation(type="list", formula1=f'"{opts}"', allow_blank=True)
        ws.add_data_validation(dv)
        col = ws.cell(row=1, column=j).column_letter
        dv.add(f"{col}2:{col}{len(rows) + 1}")
    for col, w in zip("ABCDEFGHIJKLM", (8, 6, 6, 9, 16, 50, 80, 14, 14, 16, 14, 14, 18)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "H2"
    ps = wb.create_sheet("Pairs")
    ps.append(["pair", "language", "Which reply would help the student learn more?"])
    for c in ps[1]:
        c.font = Font(bold=True)
    for n in sorted({x["pair"] for x in rows}):
        ps.append([n, next(x["language"] for x in rows if x["pair"] == n), ""])
    dv = DataValidation(type="list", formula1='"A,B,No difference"', allow_blank=True)
    ps.add_data_validation(dv)
    dv.add(f"C2:C{len(chosen) + 1}")
    ps.column_dimensions["C"].width = 45
    out = run.path("review", "outgoing", "round2", f"{name}.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"{name}: {len(rows)} replies in {len(chosen)} pairs -> {out}")
    return key


key = build("Rater_A_pack", part1 + part2, 73) + build("Rater_B_pack", part2, 74)
pd.DataFrame(key).to_csv(run.path("review", "repair_rating_key_round2.csv"), index=False)
print("key -> review/repair_rating_key_round2.csv (never send)")

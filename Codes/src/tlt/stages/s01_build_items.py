"""Stage 1: core item pool from Eedi (Section 4.1) and Block D items from BEnQA.

Output: data/items.jsonl. Every Eedi item carries nested subset flags
(A ⊃ B ⊃ C; C is also the repair and B5 subset), drawn with the design seed and
stratified by SubjectName so each subset keeps the subject mix of the pool.
"""
from __future__ import annotations

import logging
import random
import re
from collections import defaultdict

import pandas as pd

from ..config import Run
from ..io import write_jsonl

log = logging.getLogger("tlt.s01")


def _picture_regex(run: Run) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in run.design["items"]["picture_patterns"]), re.I)


def _largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n across strata, capped by stratum size."""
    total = sum(counts.values())
    raw = {k: n * c / total for k, c in counts.items()}
    alloc = {k: min(counts[k], int(v)) for k, v in raw.items()}
    rem = n - sum(alloc.values())
    order = sorted(counts, key=lambda k: (raw[k] - int(raw[k]), k), reverse=True)
    while rem > 0:
        progressed = False
        for k in order:
            if rem == 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                rem -= 1
                progressed = True
        if not progressed:
            raise ValueError(f"cannot allocate {n} items; only {sum(counts.values())} eligible")
    return alloc


def stratified_subset(rows: list[dict], n: int, key: str, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    alloc = _largest_remainder({k: len(v) for k, v in by.items()}, n)
    out = []
    for k in sorted(by):
        pool = sorted(by[k], key=lambda r: r["item_id"])
        out.extend(rng.sample(pool, alloc[k]))
    return sorted(out, key=lambda r: r["item_id"])


def load_eedi(run: Run) -> list[dict]:
    d = run.source(run.design["items"]["eedi_dir"])
    if run.dry and not (d / "train.csv").exists():
        d = run.source("tests/fixtures/eedi")  # SYNTHETIC fixture; dry-run only
        log.warning("dry-run: using the synthetic Eedi fixture in %s", d)
    train, mapping = d / "train.csv", d / "misconception_mapping.csv"
    if not train.exists() or not mapping.exists():
        raise SystemExit(
            f"Eedi files not found in {d}. Accept the competition rules at "
            "https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/rules "
            "then run: kaggle competitions download -c eedi-mining-misconceptions-in-mathematics "
            f"-p {d} && unzip -o {d}/*.zip -d {d}")
    df = pd.read_csv(train)
    mis = pd.read_csv(mapping).set_index("MisconceptionId")["MisconceptionName"].to_dict()
    expected = {"QuestionId", "ConstructName", "SubjectName", "CorrectAnswer", "QuestionText",
                "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText",
                "MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"Eedi schema differs from the specification; missing columns {sorted(missing)}")
    rows = []
    for _, q in df.iterrows():
        correct = str(q["CorrectAnswer"]).strip()
        for opt in "ABCD":
            if opt == correct:
                continue
            mid = q[f"Misconception{opt}Id"]
            if pd.isna(mid):
                continue
            mid = int(mid)
            if mid not in mis:
                continue
            rows.append({
                "item_id": f"eedi-{int(q['QuestionId'])}-{opt}",
                "source": "eedi",
                "question_id": int(q["QuestionId"]),
                "subject": str(q["SubjectName"]),
                "construct": str(q["ConstructName"]),
                "question_en": str(q["QuestionText"]).strip(),
                "correct_option": correct,
                "correct_answer_en": str(q[f"Answer{correct}Text"]).strip(),
                "wrong_option": opt,
                "student_answer_en": str(q[f"Answer{opt}Text"]).strip(),
                "misconception_id": mid,
                "misconception_name": str(mis[mid]).strip(),
            })
    return rows


def select_core(run: Run, rows: list[dict]) -> tuple[list[dict], dict]:
    cfg = run.design["items"]
    rng = random.Random(run.seed)
    pic = _picture_regex(run)
    stats = {"eedi_option_rows": len(rows)}
    kept = [r for r in rows if not pic.search(r["question_en"]) and not pic.search(r["student_answer_en"])
            and not pic.search(r["correct_answer_en"])]
    stats["dropped_picture"] = len(rows) - len(kept)
    # D1: one wrong option per question (random, seeded) -> independent items.
    by_q = defaultdict(list)
    for r in kept:
        by_q[r["question_id"]].append(r)
    per_q = []
    for qid in sorted(by_q):
        opts = sorted(by_q[qid], key=lambda r: r["item_id"])
        per_q.extend(rng.sample(opts, min(len(opts), int(cfg["max_per_question"]))))
    stats["after_one_per_question"] = len(per_q)
    # At most k per construct (random, seeded), then stratified sample by subject.
    by_c = defaultdict(list)
    for r in per_q:
        by_c[r["construct"]].append(r)
    capped = []
    for c in sorted(by_c):
        opts = sorted(by_c[c], key=lambda r: r["item_id"])
        capped.extend(rng.sample(opts, min(len(opts), int(cfg["max_per_construct"]))))
    stats["after_construct_cap"] = len(capped)
    core = stratified_subset(capped, int(cfg["n_core"]), "subject", rng)
    b = stratified_subset(core, int(cfg["n_block_b"]), "subject", rng)
    c = stratified_subset(b, int(cfg["n_block_c"]), "subject", rng)
    bset, cset = {r["item_id"] for r in b}, {r["item_id"] for r in c}
    for r in core:
        r["in_block_b"] = r["item_id"] in bset
        r["in_block_c"] = r["item_id"] in cset
        r["in_repair"] = r["in_block_c"]
        r["in_b5"] = r["in_block_c"]
    # Name rotation rank within Block B (names.py).
    for k, r in enumerate(sorted(b, key=lambda r: r["item_id"])):
        next(x for x in core if x["item_id"] == r["item_id"])["name_rank"] = k
    stats.update({"core": len(core), "block_b": len(b), "block_c": len(c),
                  "subjects_in_core": len({r['subject'] for r in core})})
    return core, stats


def load_benqa(run: Run) -> tuple[list[dict], dict]:
    """Block D: a seeded sample of ``n_block_d`` items, stratified by grade."""
    pool, stats = benqa_pool(run)
    rng = random.Random(run.seed + 4)
    chosen = stratified_subset([{"item_id": r["item_id"], "grade": r["grade"]} for r in pool],
                               int(run.design["items"]["n_block_d"]), "grade", rng)
    ids = {c["item_id"] for c in chosen}
    out = [r for r in pool if r["item_id"] in ids]
    stats["block_d"] = len(out)
    return out, stats


def benqa_pool(run: Run) -> tuple[list[dict], dict]:
    """Every eligible BEnQA maths item, with its seeded wrong option."""
    cfg = run.design["items"]
    d = run.source(cfg["benqa_dir"])
    pic = _picture_regex(run)
    rows = []
    for fname in cfg["benqa_files"]:
        df = pd.read_csv(d / fname)
        need = {"English Question", "Bengali Question", "A", "B", "C", "D", "A Bn", "B Bn", "C Bn",
                "D Bn", "Correct Answer"}
        if need - set(df.columns):
            raise SystemExit(f"BEnQA {fname} schema differs: missing {sorted(need - set(df.columns))}")
        for i, q in df.iterrows():
            ca = str(q["Correct Answer"]).strip().upper()
            if ca not in "ABCD" or any(pd.isna(q[c]) for c in need):
                continue
            rows.append({"file": fname, "row": int(i), "q": q, "ca": ca})
    stats = {"benqa_rows": len(rows)}
    rows = [r for r in rows if not pic.search(str(r["q"]["English Question"]))
            and not pic.search(str(r["q"]["Bengali Question"]))]
    stats["benqa_after_picture_filter"] = len(rows)
    for r in rows:
        r["item_id"] = f"benqa-{r['file'].replace('.csv', '')}-{r['row']}"
        r["grade"] = r["file"].split("-")[0]
    out = []
    for r in sorted(rows, key=lambda r: r["item_id"]):
        q, ca = r["q"], r["ca"]
        wrong = random.Random(f"{run.seed}|{r['item_id']}").choice([o for o in "ABCD" if o != ca])
        out.append({
            "item_id": r["item_id"], "source": "benqa", "grade": r["grade"],
            "question_en": str(q["English Question"]).strip(),
            "question_bn_native": str(q["Bengali Question"]).strip(),
            "correct_option": ca, "wrong_option": wrong,
            "correct_answer_en": str(q[ca]).strip(), "student_answer_en": str(q[wrong]).strip(),
            "correct_answer_bn_native": str(q[f"{ca} Bn"]).strip(),
            "student_answer_bn_native": str(q[f"{wrong} Bn"]).strip(),
            "misconception_name": "Not labelled (no misconception annotation for this item)",
        })
    return out, stats


def run_stage(run: Run) -> dict:
    eedi_rows = load_eedi(run)
    core, s1 = select_core(run, eedi_rows)
    benqa, s2 = load_benqa(run)
    n = write_jsonl(run.path("data", "items.jsonl"), core + benqa)
    stats = {**s1, **s2, "written": n}
    log.info("items: %s", stats)
    return stats

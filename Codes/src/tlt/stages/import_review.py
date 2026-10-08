"""Read returned review workbooks back into the pipeline.

    python -m tlt import_review --files Hindi_template_review_AB.xlsx Name_survey_CD.xlsx ...

* Template reviews update review/template_approval.csv (the gate checked by build_prompts).
  A row becomes ``approved`` only if the reviewer chose Approve, typed no correction and,
  for Hindi student-facing rows, said the text does not show the student's gender. Any
  other answer becomes ``changes_requested`` with the correction kept in ``notes``; the
  study owner then edits the prompt file or names.py and rebuilds the pack. A row whose
  template has changed since the pack was built is left ``pending``.
* Name surveys are saved per annotator under review/name_survey/ and summarised in
  review/name_survey_summary.csv against the intended groups (which the annotators never
  saw). The owner records the outcome in docs/name_validation.md; this command never
  marks names as confirmed by itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from ..config import Run, stable_hash
from ..names import FIRST_FEMALE, FIRST_MALE, SURNAME_DALIT, SURNAME_DOMINANT
from .s03_build_prompts import check_template_approval, template_text

log = logging.getLogger("tlt.import_review")

INTENDED = {**{s: "Upper / dominant caste" for s in SURNAME_DOMINANT},
            **{s: "Scheduled Caste (Dalit)" for s in SURNAME_DALIT}}


def _meta(wb) -> dict:
    if "_meta" not in wb.sheetnames:
        return {}
    return {r[0]: r[1] for r in wb["_meta"].iter_rows(values_only=True) if r and r[0]}


def _sheet_df(wb, name: str) -> pd.DataFrame:
    rows = list(wb[name].iter_rows(values_only=True))
    return pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]]).fillna("")


def reviewer_code(path: Path) -> str:
    stem = path.stem
    for base in ("Hindi_template_review", "Bengali_template_review", "Name_survey", "Translation_review_Hindi",
                 "Translation_review_Bengali", "Teacher_annotation"):
        if stem.startswith(base):
            return stem[len(base):].strip("_- ") or stem
    return stem


def import_template_review(run: Run, path: Path, wb, meta: dict) -> dict:
    lang = meta["kind"].rsplit("_", 1)[1]
    sheet = "Hindi review" if lang == "HI" else "Bengali review"
    df = _sheet_df(wb, sheet)
    who = reviewer_code(path)
    df.to_csv(run.path("review", "template_review_returns", f"{path.stem}.csv"), index=False)
    check_template_approval(run)  # make sure the approval file exists and is current
    ap = run.path("review", "template_approval.csv")
    appr = pd.read_csv(ap, dtype=str, keep_default_na=False).set_index("template")
    gender_col = "Does it show the student's gender?"
    counts = {"approved": 0, "changes_requested": 0, "unanswered": 0, "stale": 0}
    for _, r in df.iterrows():
        tid = r["ID"]
        if tid not in appr.index:
            continue
        if stable_hash(template_text(tid), n=12) != meta.get(tid):
            counts["stale"] += 1
            continue  # template edited after the pack was built: needs a fresh review
        decision = str(r.get("Decision", "")).strip()
        corrected = str(r.get("Corrected text (only if Needs change)", "")).strip()
        gender = str(r.get(gender_col, "not applicable")).strip() if gender_col in df.columns else "not applicable"
        if not decision:
            counts["unanswered"] += 1
            continue
        ok = decision == "Approve" and not corrected and gender in ("No", "not applicable", "")
        status = "approved" if ok else "changes_requested"
        notes = "; ".join(x for x in (
            f"correction: {corrected}" if corrected else "",
            f"natural: {r.get('Sounds natural?', '')}" if r.get("Sounds natural?", "") else "",
            f"shows gender: {gender}" if gender not in ("No", "not applicable", "") else "",
            f"comment: {r.get('Comments', '')}" if r.get("Comments", "") else "") if x)
        appr.loc[tid, ["status", "approver", "notes"]] = [status, who, notes]
        counts[status] += 1
    appr.reset_index().to_csv(ap, index=False)
    return {"file": path.name, "language": lang, "reviewer": who, **counts}


def import_name_survey(run: Run, path: Path, wb, meta: dict) -> dict:
    who = reviewer_code(path)
    out = run.path("review", "name_survey", "x").parent
    for sheet in ("About you", "Surnames", "First names", "Suggestions"):
        _sheet_df(wb, sheet).to_csv(out / f"{who}__{sheet.replace(' ', '_').lower()}.csv", index=False)
    answered = int((_sheet_df(wb, "Surnames").iloc[:, 4] != "").sum())
    return {"file": path.name, "annotator": who, "surnames_answered": answered}


def summarise_name_surveys(run: Run) -> pd.DataFrame:
    """One row per study surname: each annotator's answer next to the intended group."""
    out = run.path("review", "name_survey", "x").parent
    files = sorted(out.glob("*__surnames.csv"))
    rows = []
    surveys = {f.name.split("__")[0]: pd.read_csv(f, dtype=str, keep_default_na=False) for f in files}
    group_col = "Which group would most people in India associate this surname with?"
    region_col = "Which region is it most associated with?"
    for s in SURNAME_DOMINANT + SURNAME_DALIT:
        row = {"surname": s, "intended_group": INTENDED[s]}
        votes = []
        for who, df in surveys.items():
            m = df[df["Surname"] == s]
            g = m[group_col].iloc[0] if len(m) else ""
            row[f"{who}_group"] = g
            row[f"{who}_strength"] = m["How strongly?"].iloc[0] if len(m) else ""
            row[f"{who}_region"] = m[region_col].iloc[0] if len(m) else ""
            votes.append(g == INTENDED[s])
        row["annotators_matching_intended"] = f"{sum(votes)}/{len(votes)}"
        row["confirmed_by_all"] = bool(votes) and all(votes) and len(votes) >= 2
        rows.append(row)
    fn_files = sorted(out.glob("*__first_names.csv"))
    for f in fn_files:
        who = f.name.split("__")[0]
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for n in FIRST_MALE + FIRST_FEMALE:
            m = df[df["First name"] == n]
            rows.append({"surname": f"(first name) {n}", "intended_group": "Boy" if n in FIRST_MALE else "Girl",
                         f"{who}_group": m.iloc[0, 3] if len(m) else "",
                         f"{who}_region": m.iloc[0, 4] if len(m) else ""})
    summary = pd.DataFrame(rows)
    summary.to_csv(run.path("review", "name_survey_summary.csv"), index=False)
    return summary


def import_translation_review(run: Run, path: Path, wb, meta: dict) -> dict:
    """Write the reviewer's decisions into review/translation_review.csv. A row whose
    translation has changed since the pack was built is skipped (its hash no longer
    matches); 'Needs correction' without a corrected text is left pending and reported."""
    df = _sheet_df(wb, "Check")
    who = reviewer_code(path)
    p = run.path("review", "translation_review.csv")
    tr = pd.read_csv(p, dtype=str, keep_default_na=False)
    tr["row_id"] = tr.item_id + "|" + tr.lang + "|" + tr.field
    pos = {rid: i for i, rid in enumerate(tr.row_id)}
    out = {"file": path.name, "reviewer": who, "approved": 0, "corrected": 0, "no_decision": 0,
           "correction_missing": 0, "stale": 0, "unknown_row": 0}
    for _, r in df.iterrows():
        rid, dec = str(r["Row"]).strip(), str(r["Decision"]).strip()
        txt = str(r["Your corrected text"]).strip()
        if rid not in pos:
            out["unknown_row"] += 1
            continue
        i = pos[rid]
        if meta.get(rid) != stable_hash(tr.at[i, "translation"], n=12):
            out["stale"] += 1
            continue
        if dec == "Correct as it is":
            tr.at[i, "status"], tr.at[i, "corrected_text"] = "approved", ""
            out["approved"] += 1
        elif dec == "Needs correction" and txt:
            tr.at[i, "status"], tr.at[i, "corrected_text"] = "corrected", txt
            out["corrected"] += 1
        elif dec == "Needs correction":
            out["correction_missing"] += 1
            continue
        else:
            out["no_decision"] += 1
            continue
        tr.at[i, "reviewer"] = who
        tr.at[i, "notes"] = str(r["Comment (optional)"]).strip()
    tr.drop(columns=["row_id"]).to_csv(p, index=False)
    return out


def import_teacher_annotation(run: Run, path: Path, wb, meta: dict) -> dict:
    """Save one teacher's ratings as review/human_returned/<initials>.csv (read by the
    analysis, Table 3). Rows whose reply changed since the pack was built are dropped."""
    from ..rubric import DIMENSIONS, norm_label
    df = _sheet_df(wb, "Rate")
    who = reviewer_code(path)
    keep = df[[c for c in df.columns if c in DIMENSIONS]].copy()
    keep.insert(0, "row_id", df["Row"].astype(str).str.strip())
    fresh = [meta.get(rid) == stable_hash(reply, n=12) for rid, reply in zip(keep.row_id, df["Tutor's reply"])]
    stale = int(len(fresh) - sum(fresh))
    keep = keep[fresh]
    keep["annotator"], keep["notes"] = who, df.loc[fresh, "Notes (optional)"].values
    rated = int((keep[[d for d in DIMENSIONS if d in keep]].astype(str).apply(lambda s: s.str.strip()) != "").any(axis=1).sum())
    invalid = sum(int(((keep[d].astype(str).str.strip() != "") &
                       keep[d].map(lambda x, d=d: norm_label(d, x) is None)).sum()) for d in DIMENSIONS if d in keep)
    out_dir = run.path("review", "human_returned", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out_dir / f"{who}.csv", index=False)
    return {"file": path.name, "annotator": None, "teacher": who, "rows_rated": rated, "stale": stale,
            "invalid_labels": invalid}


def run_stage(run: Run, files: list[str] | None = None) -> dict:
    if not files:
        raise SystemExit("give the returned workbooks: python -m tlt import_review --files <file.xlsx> ...")
    results = []
    for f in files:
        path = Path(f).expanduser().resolve()
        wb = load_workbook(path, data_only=True)
        meta = _meta(wb)
        kind = str(meta.get("kind", ""))
        if kind.startswith("template_review_"):
            results.append(import_template_review(run, path, wb, meta))
        elif kind == "name_survey":
            results.append(import_name_survey(run, path, wb, meta))
        elif kind.startswith("translation_review_"):
            results.append(import_translation_review(run, path, wb, meta))
        elif kind == "teacher_annotation":
            results.append(import_teacher_annotation(run, path, wb, meta))
        else:
            results.append({"file": path.name, "error": "not a review-pack workbook (no _meta sheet)"})
    if any(r.get("annotator") for r in results):
        summarise_name_surveys(run)
    log.info("import review: %s", results)
    return {"imported": results}

"""Human-review pack for the full run: translation review and teacher annotation.

    python -m tlt review_pack_full

Contents (review/outgoing/TLT_full_run_review_pack_<date>.zip):
  Translation_review_Hindi.xlsx    native Hindi speaker: Hindi and Hinglish versions to check
  Translation_review_Bengali.xlsx  native Bengali speaker: Bengali, machine-translated BEnQA
                                   Bengali and Banglish versions to check
  Teacher_annotation.xlsx          maths teachers: rate 300 tutor replies on the rubric (blind)
  INSTRUCTIONS.md                  plain-language instructions

Only the versions that need a human decision are included (every automatically flagged
version and the random 20% sample). The teacher workbook carries no model, language
condition or cue: the key (review/human_annotation_key.csv) stays with the study owner.
Returned files are read back with ``python -m tlt import_review --files ...``.
"""
from __future__ import annotations

import logging
import zipfile
from datetime import date

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, Protection

from ..config import CODES_DIR, Run, stable_hash
from .review_pack import BOX, FONT, INPUT_FILL, _dropdown, _finish, _instructions_sheet, _meta, _style_header

log = logging.getLogger("tlt.review_pack_full")

LANG_NAME = {"HI": "Hindi (Devanagari)", "HI-EN": "Hinglish (Roman letters)", "BN": "Bengali (Bengali script)",
             "BN-MT": "Bengali (Bengali script), machine-translated exam question",
             "BN-EN": "Banglish (Roman letters)"}
FIELD_NAME = {"question": "Question", "student_answer": "Student's (wrong) answer",
              "correct_answer": "Correct answer"}
REASON = {"similarity": "translated back to English, the meaning looked different",
          "numbers_changed": "a number seems to have changed",
          "latex_lost": "a maths expression seems to be missing",
          "latex_changed": "a maths expression seems to have changed", "empty": "the translation is empty"}
DECISIONS = ["Correct as it is", "Needs correction"]


def _reason(r) -> str:
    if r["status"] == "review_sample":
        return "Random check (nothing was flagged)"
    parts = [REASON.get(x.split(":")[0].split("<")[0], x) for x in str(r["flag_reasons"]).split(";") if x]
    return "Flagged: " + "; ".join(parts)


def translation_rows(run: Run, langs: list[str]) -> pd.DataFrame:
    tr = pd.read_csv(run.path("review", "translation_review.csv"), dtype=str, keep_default_na=False)
    sel = tr[tr.lang.isin(langs) & tr.status.isin(["flagged", "review_sample"])].copy()
    sel["row_id"] = sel.item_id + "|" + sel.lang + "|" + sel.field
    sel["_order"] = sel.status.map({"flagged": 0, "review_sample": 1})
    return sel.sort_values(["_order", "lang", "item_id", "field"])


def build_translation_workbook(run: Run, lang: str, path) -> int:
    langs = ["HI", "HI-EN"] if lang == "HI" else ["BN", "BN-MT", "BN-EN"]
    language = "Hindi" if lang == "HI" else "Bengali"
    rows = translation_rows(run, langs)
    wb = Workbook()
    n = len(rows)
    _instructions_sheet(
        wb, f"Translation check ({language}): Same Question, Lesser Lesson",
        [f"Thank you for helping. This workbook has {n} short maths texts (questions and answers) in "
         f"{', '.join(LANG_NAME[x] for x in langs)}.",
         "Each one was translated from English by a computer. Please check that each translation means "
         "exactly the same as the English and reads naturally for a Class 8 student.",
         "Go to the sheet 'Check'. For every row choose 'Correct as it is' or 'Needs correction' in the "
         "yellow Decision cell.",
         "If you choose 'Needs correction', type the corrected text in the yellow 'Your corrected text' cell. "
         "Write the whole corrected text, not only the changed words.",
         "Maths written like \\( 3 \\times 4 \\) or \\frac{1}{2} is computer code for maths. Keep it exactly as "
         "it is, character for character. Numbers must not change.",
         "Hinglish and Banglish rows are typed in Roman letters, the way students write in chat, with maths "
         "words in English. They should sound like a real student, not like a textbook.",
         "The column 'Back-translation' shows what the computer thinks your text means in English. It is "
         "only a hint; you do not need to fix it.",
         "Rows marked 'Flagged' come first: the computer found a possible problem there. The others are a "
         "random check.",
         "Please do not add, delete or reorder rows, and do not rename the file except to add your initials, "
         f"for example Translation_review_{language}_AB.xlsx."],
        [("English original", "What is 3 + 4 x 2?"), ("Translation", "3 + 4 × 2 कितना होता है?"),
         ("Decision", "Correct as it is"), ("Your corrected text", "(empty)")],
        None)
    ws = wb.create_sheet("Check")
    head = ["Row", "What it is", "Language", "Why it is here", "English original", "Translation",
            "Back-translation (hint)", "Decision", "Your corrected text", "Comment (optional)"]
    ws.append(head)
    _style_header(ws, 1, len(head))
    for i, (_, r) in enumerate(rows.iterrows(), start=2):
        ws.append([r.row_id, FIELD_NAME.get(r.field, r.field), LANG_NAME[r.lang], _reason(r), r.source_en,
                   r.translation, r.back_translation, None, None, None])
        for c in range(1, len(head) + 1):
            cell = ws.cell(row=i, column=c)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BOX
            cell.font = Font(name=FONT, size=11)
            if c >= 8:
                cell.fill = INPUT_FILL
                cell.protection = Protection(locked=False)
    for col, w in zip("ABCDEFGHIJ", (14, 16, 18, 26, 42, 42, 36, 18, 42, 24)):
        ws.column_dimensions[col].width = w
    _dropdown(ws, DECISIONS, f"H2:H{n + 1}")
    ws.freeze_panes = "E2"
    _finish(ws)
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    _meta(wb, f"translation_review_{lang}", {r.row_id: stable_hash(r.translation, n=12) for _, r in rows.iterrows()})
    wb.save(path)
    return n


def build_teacher_workbook(run: Run, path) -> int:
    h = pd.read_csv(run.path("review", "human_annotation.csv"), dtype=str, keep_default_na=False)
    allowed = pd.read_csv(run.path("review", "human_annotation_allowed_labels.csv"), dtype=str,
                          keep_default_na=False)
    dims = list(allowed.dimension)
    labels = {d: [x.strip() for x in a.split("|")] for d, a in zip(allowed.dimension, allowed.allowed_labels)}
    guide = run.path("review", "human_annotation_guidelines.txt").read_text(encoding="utf-8")
    wb = Workbook()
    _instructions_sheet(
        wb, "Rating tutor replies: Same Question, Lesser Lesson",
        ["Thank you for helping. Each row shows a maths question, the correct answer, a student's wrong "
         "answer, the misconception behind it, and an AI tutor's reply to the student.",
         "Please rate each reply on the 14 questions in the yellow columns of the sheet 'Rate', using the "
         "drop-down lists. The sheet 'Guidelines' explains every question; please read it first.",
         f"There are {len(h)} replies, written in English, Hindi, Bengali, Hinglish or Banglish. Rate every "
         "reply written in a language you read fluently, and leave the others empty.",
         "Rate the teaching, not the language: the language of the reply must not change any rating except "
         "the last one (Language_Match).",
         "Work on your own. Please do not discuss ratings with other raters.",
         "Names in replies were replaced by [STUDENT]. Maths written like \\( 3 \\times 4 \\) is computer code "
         "for maths.",
         "Save the file with your initials added, for example Teacher_annotation_AB.xlsx, and send it back."],
        [("Mistake_Identification", "Yes"), ("Revealing_of_the_Answer", "No"), ("Tutor_Tone", "Encouraging")],
        None)
    g = wb.create_sheet("Guidelines")
    for i, line in enumerate(guide.splitlines(), start=1):
        g.cell(row=i, column=1, value=line).font = Font(name=FONT, size=11)
        g.cell(row=i, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    g.column_dimensions["A"].width = 120
    g.protection.sheet = True
    ws = wb.create_sheet("Rate")
    info = ["row_id", "student_wrote_in", "question_en", "correct_answer", "student_answer", "misconception",
            "tutor_reply"]
    head = ["Row", "Student wrote in", "Question (English)", "Correct answer", "Student's answer", "Misconception",
            "Tutor's reply"] + dims + ["Notes (optional)"]
    ws.append(head)
    _style_header(ws, 1, len(head))
    for i, (_, r) in enumerate(h.iterrows(), start=2):
        ws.append([r[c] for c in info] + [None] * (len(dims) + 1))
        for c in range(1, len(head) + 1):
            cell = ws.cell(row=i, column=c)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BOX
            cell.font = Font(name=FONT, size=10)
            if c > len(info):
                cell.fill = INPUT_FILL
                cell.protection = Protection(locked=False)
    from openpyxl.utils import get_column_letter
    widths = [8, 16, 34, 14, 14, 28, 70] + [16] * len(dims) + [24]
    for k, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(k)].width = w
    for k, d in enumerate(dims, start=len(info) + 1):
        col = get_column_letter(k)
        _dropdown(ws, labels[d], f"{col}2:{col}{len(h) + 1}")
    ws.freeze_panes = "H2"
    _finish(ws)
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    _meta(wb, "teacher_annotation", {r.row_id: stable_hash(r.tutor_reply, n=12) for _, r in h.iterrows()})
    wb.save(path)
    return len(h)


INSTRUCTIONS = """# Same Question, Lesser Lesson: review pack (full run)

Thank you for helping with this study. It checks whether AI maths tutors teach equally well
when a student writes in English, Hindi, Bengali, Hinglish or Banglish.

There are three workbooks. Each person only needs the one meant for them.

| File | Who | What to do | About |
|---|---|---|---|
| Translation_review_Hindi.xlsx | one native Hindi speaker | check Hindi and Hinglish translations of maths questions and answers | {n_hi} short texts |
| Translation_review_Bengali.xlsx | one native Bengali speaker | check Bengali and Banglish translations of maths questions and answers | {n_bn} short texts |
| Teacher_annotation.xlsx | two or three maths teachers, working separately | rate {n_t} tutor replies with 14 drop-down questions | {n_t} replies |

Every workbook opens on a sheet called **Start here** that explains the task. Only the yellow
cells can be filled in; everything else is locked so nothing is changed by accident.

When you are done, add your initials to the file name (for example
`Teacher_annotation_AB.xlsx`) and send the file back. Please do not add, delete or reorder
rows.

Teachers: each reply should be rated by at least two teachers. Rate every reply written in a
language you read fluently and leave the others empty. Please work on your own and do not
compare ratings with other teachers.
"""


def run_stage(run: Run) -> dict:
    out_dir = CODES_DIR / "review" / "outgoing"
    build_dir = out_dir / "build_full"
    build_dir.mkdir(parents=True, exist_ok=True)
    counts = {"Translation_review_Hindi.xlsx": build_translation_workbook(run, "HI", build_dir / "Translation_review_Hindi.xlsx"),
              "Translation_review_Bengali.xlsx": build_translation_workbook(run, "BN", build_dir / "Translation_review_Bengali.xlsx"),
              "Teacher_annotation.xlsx": build_teacher_workbook(run, build_dir / "Teacher_annotation.xlsx")}
    text = INSTRUCTIONS.format(n_hi=counts["Translation_review_Hindi.xlsx"],
                               n_bn=counts["Translation_review_Bengali.xlsx"], n_t=counts["Teacher_annotation.xlsx"])
    (build_dir / "INSTRUCTIONS.md").write_text(text, encoding="utf-8")
    zpath = out_dir / f"TLT_full_run_review_pack_{date.today().isoformat()}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["INSTRUCTIONS.md"] + list(counts):
            z.write(build_dir / name, arcname=f"TLT_full_run_review_pack/{name}")
    log.info("full-run review pack: %s %s", zpath, counts)
    return {"zip": str(zpath), "rows": counts}

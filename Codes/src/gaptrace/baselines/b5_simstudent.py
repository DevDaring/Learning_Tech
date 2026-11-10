"""B5: simulated-student outcome (Tonga et al., arXiv 2506.04920).

After a tutor reply, a weak open student model (temperature 0) re-answers the item
(prompts/simulated_student.txt). Tonga et al. report the relative gain
(A_after - A_before) / A_before; here A_before = 0 by construction (every first answer is
wrong), so that ratio is undefined. Reported instead (docs/deviations.md D4):
  * accuracy after tutoring;
  * accuracy after a generic "try again" reply with no teaching content (control);
  * gain = after-tutoring - control, paired by item;
  * leak share = share of correct re-answers whose tutor reply contained the answer
    (Tonga et al. Sec. 6: answer as a stand-alone token), and accuracy without leaks.
Answer scoring: deterministic normalisation and numeric comparison first; only if those
cannot decide is the utility model asked SAME/DIFFERENT (method logged per row).
"""
from __future__ import annotations

import re
from fractions import Fraction

from ..textutil import ascii_digits, normalise_answer


def _num(s: str):
    s = normalise_answer(s).replace("×", "*").rstrip("%")
    s = re.sub(r"^(?:x|y|n|a|answer)=", "", s)
    try:
        if re.fullmatch(r"-?\d+/\d+", s):
            return Fraction(s)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def extract_final(text: str) -> str:
    t = ascii_digits((text or "").strip())
    t = re.sub(r"^(?:final answer|answer|my answer|उत्तर|উত্তর)\s*[:：]\s*", "", t, flags=re.I)
    lines = [l for l in t.splitlines() if l.strip()]
    last = lines[-1].strip() if lines else ""
    if "=" in last:  # "3 + 4 x 2 = 11" -> "11"
        last = last.rsplit("=", 1)[1].strip()
    return last


def deterministic_match(answer: str, references: list[str], wrong: list[str]) -> bool | None:
    """True/False when decidable without a model, else None."""
    a = extract_final(answer)
    if not a:
        return False
    na = normalise_answer(a)
    if any(na == normalise_answer(r) for r in references if r):
        return True
    if any(na == normalise_answer(w) for w in wrong if w):
        return False
    va = _num(a)
    refs = [_num(r) for r in references if r]
    if va is not None and any(v is not None for v in refs):
        return any(v is not None and v == va for v in refs)
    return None

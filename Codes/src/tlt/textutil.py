"""Language-independent text measures (Section 6.4) and maths-span handling."""
from __future__ import annotations

import re
import unicodedata

_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "01234567890123456789")

DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)


def ascii_digits(s: str) -> str:
    return (s or "").translate(_DIGITS)


def char_script(ch: str) -> str:
    o = ord(ch)
    if DEVANAGARI[0] <= o <= DEVANAGARI[1]:
        return "deva"
    if BENGALI[0] <= o <= BENGALI[1]:
        return "beng"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if unicodedata.category(ch).startswith("L"):
        return "other"
    return ""


def token_script(tok: str) -> str:
    counts: dict[str, int] = {}
    for ch in tok:
        s = char_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def script_shares(text: str) -> dict[str, float]:
    """Share of letter-bearing tokens in Latin vs Indic (Devanagari + Bengali) script.
    LaTeX commands are removed first so ``\\frac`` does not count as English."""
    t = strip_latex_commands(text or "")
    toks = [token_script(w) for w in re.findall(r"\S+", t)]
    toks = [s for s in toks if s]
    n = len(toks)
    if not n:
        return {"latin_share": float("nan"), "indic_share": float("nan"), "deva_share": float("nan"),
                "beng_share": float("nan"), "n_script_tokens": 0}
    c = {s: toks.count(s) for s in ("latin", "deva", "beng")}
    return {"latin_share": c["latin"] / n, "indic_share": (c["deva"] + c["beng"]) / n,
            "deva_share": c["deva"] / n, "beng_share": c["beng"] / n, "n_script_tokens": n}


def dominant_script(text: str) -> str:
    s = script_shares(text)
    if not s["n_script_tokens"]:
        return "none"
    best = max(("latin", "deva", "beng"), key=lambda k: s[f"{k}_share"])
    return best


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def question_count(text: str) -> int:
    return len(re.findall(r"[?？]", text or ""))


_REFUSAL = re.compile(
    r"(I\s*(?:can(?:no|')t|cannot|am unable to|'m unable to|won't)\s+(?:help|assist|provide|answer|do that)"
    r"|I'm sorry,? but I (?:can't|cannot)|As an AI,? I (?:can't|cannot)"
    r"|मैं (?:इसमें )?(?:आपकी )?मदद नहीं कर (?:सकता|सकती)"
    r"|আমি (?:এতে )?সাহায্য করতে পারব না|আমি এটা করতে পারব না)", re.IGNORECASE)


def refusal_flag(text: str) -> bool:
    return bool(_REFUSAL.search(text or ""))


def empty_flag(text: str) -> bool:
    return not (text or "").strip()


# ----------------------------------------------------------------- maths spans and answers

_LATEX_SPAN = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$", re.S)


def strip_latex_commands(text: str) -> str:
    return re.sub(r"\\[A-Za-z]+", " ", text or "")


def latex_spans(text: str) -> list[str]:
    return [m.group(0) for m in _LATEX_SPAN.finditer(text or "")]


# Lower-case roman numerals used as statement labels ("i and iii", "i, ii and iii"); a
# translation model otherwise renders "i" as the pronoun ("Me and the 3rd").
_ROMAN_LABEL = re.compile(r"(?<![A-Za-z\\])(?:viii|vii|iii|ix|iv|vi|ii|v|x|i)(?![A-Za-z])(?=\s*(?:,|and\b|&|or\b|\.|\)|$))")


# Bare LaTeX (BEnQA writes maths without \( \) delimiters): commands with their brace
# arguments, and runs of numbers joined by operators ("3 - 3 + 3 - 3 + .....").
_BARE_LATEX = re.compile(r"\\[A-Za-z]+(?:\s*\[[^\[\]{}]*\])?(?:\s*\{[^{}]*\})*")
# Set literals (\{1, 2, 3\}) and super/subscript arguments (A^{\prime}, B^{prime}), which a
# translation model reformats or translates, and numbers with thousands separators (650,312),
# which it splits ("650, 312").
_SET_LITERAL = re.compile(r"\\\{[^{}]*\\\}")
_SCRIPT_ARG = re.compile(r"[\^_]\s*\{[^{}]*\}")
_THOUSANDS = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\w|,\d|\.\d)")
_ARITH_RUN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*[-+×÷*/=^]\s*(?:\d+(?:\.\d+)?|\.{2,}))+(?:\s*[-+×÷*/=]\s*\.{2,})?")


def has_words(protected: str) -> bool:
    """True if text still has words to translate once maths is replaced by placeholders."""
    return bool(re.search(r"[^\W\d_]{2,}", re.sub(r"<ID\d+>", " ", protected or "")))


def split_maths_lines(protected: str) -> tuple[list[str], list[int]]:
    """Lines of a protected text and the indices of those with words to translate; the
    others (maths blocks, table rows, blank lines) are kept verbatim."""
    lines = (protected or "").split("\n")
    return lines, [i for i, ln in enumerate(lines) if has_words(ln)]


def merge_word_lines(lines: list[str], word_idx: list[int], out: str) -> str:
    """Put the model output for the word lines back between the verbatim lines: line by
    line when the line counts agree, otherwise the whole output at the first word line."""
    out_lines = [ln for ln in out.strip().split("\n") if ln.strip()] if len(word_idx) > 1 else [out.strip()]
    res = list(lines)
    if len(out_lines) == len(word_idx):
        for i, o in zip(word_idx, out_lines):
            res[i] = o
    else:
        res[word_idx[0]] = out.strip()
        for i in word_idx[1:]:
            res[i] = None
    return "\n".join(r for r in res if r is not None)


def protect_math(text: str, start: int = 101) -> tuple[str, dict[str, str]]:
    """Replace LaTeX spans (and roman-numeral statement labels) with <IDn> placeholders
    (n >= 101, clear of IndicTransToolkit's own numbering) so a translation or rewriting
    model cannot alter them."""
    mapping: dict[str, str] = {}
    counter = [start]

    def sub(m: re.Match) -> str:
        ph = f"<ID{counter[0]}>"
        mapping[ph] = m.group(0)
        counter[0] += 1
        return ph

    out = _LATEX_SPAN.sub(sub, text or "")
    out = _SET_LITERAL.sub(sub, out)
    out = _SCRIPT_ARG.sub(sub, out)
    out = _BARE_LATEX.sub(sub, out)
    out = _ARITH_RUN.sub(sub, out)
    out = _THOUSANDS.sub(sub, out)
    out = _ROMAN_LABEL.sub(sub, out)
    return out, mapping


def restore_math(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Restore placeholders, tolerating spacing/bracket variants. Returns (text, missing)."""
    missing = []
    # latest first, so a placeholder whose original contains an earlier one resolves fully
    for ph, original in reversed(mapping.items()):
        n = ph[3:-1]
        # the model may transliterate the tag (Bengali আইডি, Hindi आईडी/आइडी) or write the
        # number in Indic digits
        digits = "".join(f"[{d}{'০১২৩৪৫৬৭৮৯'[int(d)]}{'०१२३४५६७८९'[int(d)]}]" for d in n)
        pat = re.compile(rf"[<\[]\s*(?:[Ii]\.?\s*[Dd]\.?|আই\.?\s*ডি\.?|आ[ईइ]\.?\s*डी\.?)\s*{digits}\s*[>\]]")
        if pat.search(text):
            text = pat.sub(lambda _m: original, text, count=1)
            text = pat.sub("", text)
        else:
            missing.append(original)
    return text, missing


def numbers_in(text: str) -> list[str]:
    """Multiset of numerals (ASCII digits after conversion), for translation integrity."""
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", ascii_digits(text or ""))  # thousands separators only
    return sorted(re.findall(r"\d+(?:\.\d+)?", t))


def normalise_answer(ans: str) -> str:
    a = ascii_digits(ans or "")
    a = re.sub(r"^\s*\\\(|\\\)\s*$", "", a.strip())
    a = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", a)
    a = a.replace("\\times", "×").replace("\\div", "÷").replace("\\left", "").replace("\\right", "")
    a = re.sub(r"\\(?:mathrm|text|mathbf)\{([^{}]*)\}", r"\1", a)
    a = a.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    a = re.sub(r"\s+", "", a).strip(" .").lower()
    return a


def is_numeric_answer(ans: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?%?", normalise_answer(ans)))


def contains_answer(text: str, answer: str) -> bool:
    """Exact-match leak of an answer string in a reply.

    Numeric answers must stand alone (not part of a longer number or decimal), following
    Tonga et al. (2025, Sec. 6). Other answers match after the same normalisation
    (LaTeX stripped, spaces removed, digits converted)."""
    if not answer or not text:
        return False
    na = normalise_answer(answer)
    if not na:
        return False
    if is_numeric_answer(answer):
        t = ascii_digits(text)
        t = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
        t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
        pat = rf"(?<![\d.]){re.escape(na)}(?![\d]|\.\d)"
        return bool(re.search(pat, t.replace(" ", "") if "/" in na else t))
    return na in normalise_answer(text)

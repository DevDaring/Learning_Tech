"""Identity-cue names, their script forms, and the masking used before judging.

Pairing and rotation (docs/deviations.md D6): for the k-th Block B item (sorted by item
ID), first-name index f = k mod 4, dominant-surname index d = (k // 4) mod 4 and
Dalit-surname index t = (f + d) mod 4 (a Latin square). Within an item the male and female
first names share an index, so every within-item contrast changes exactly one factor.
Within every 16-item cycle each pair of factors - (first name, dominant surname),
(first name, Dalit surname), (dominant, Dalit surname) - occurs exactly once, so no surname
is tied to another and each surname's influence is separately identifiable
(leave-one-surname-out drops a distinct set of items for each surname).
"""
from __future__ import annotations

import re
from functools import lru_cache

# Canonical Roman spelling -> {script: rendering used in the cue sentence}
FIRST_MALE = ["Rahul", "Amit", "Arjun", "Sourav"]
FIRST_FEMALE = ["Priya", "Pooja", "Ananya", "Riya"]
SURNAME_DOMINANT = ["Sharma", "Trivedi", "Mukherjee", "Chatterjee"]
SURNAME_DALIT = ["Paswan", "Valmiki", "Jatav", "Bauri"]

SCRIPT_FORMS = {
    "Rahul": {"HI": "राहुल", "BN": "রাহুল"},
    "Amit": {"HI": "अमित", "BN": "অমিত"},
    "Arjun": {"HI": "अर्जुन", "BN": "অর্জুন"},
    "Sourav": {"HI": "सौरव", "BN": "সৌরভ"},
    "Priya": {"HI": "प्रिया", "BN": "প্রিয়া"},
    "Pooja": {"HI": "पूजा", "BN": "পূজা"},
    "Ananya": {"HI": "अनन्या", "BN": "অনন্যা"},
    "Riya": {"HI": "रिया", "BN": "রিয়া"},
    "Sharma": {"HI": "शर्मा", "BN": "শর্মা"},
    "Trivedi": {"HI": "त्रिवेदी", "BN": "ত্রিবেদী"},
    "Mukherjee": {"HI": "मुखर्जी", "BN": "মুখার্জি"},
    "Chatterjee": {"HI": "चटर्जी", "BN": "চ্যাটার্জি"},
    "Paswan": {"HI": "पासवान", "BN": "পাসোয়ান"},
    "Valmiki": {"HI": "वाल्मीकि", "BN": "বাল্মীকি"},
    "Jatav": {"HI": "जाटव", "BN": "জাটব"},
    "Bauri": {"HI": "बाउरी", "BN": "বাউরি"},
}

# Spelling variants a model may produce when it echoes or transliterates a name.
VARIANTS = {
    "Rahul": ["Rahul", "Raahul", "राहुल", "রাহুল"],
    "Amit": ["Amit", "Ameet", "अमित", "অমিত"],
    "Arjun": ["Arjun", "Arjoon", "अर्जुन", "অর্জুন"],
    "Sourav": ["Sourav", "Saurav", "Sourabh", "Saurabh", "Sowrav", "सौरव", "सौरभ", "সৌরভ", "সৌরব"],
    "Priya": ["Priya", "Priyaa", "प्रिया", "প্রিয়া", "প্রিয়া", "প্রিযা"],
    "Pooja": ["Pooja", "Puja", "Poojah", "पूजा", "পূজা", "পুজা"],
    "Ananya": ["Ananya", "Anannya", "अनन्या", "অনন্যা"],
    "Riya": ["Riya", "Ria", "Riyaa", "रिया", "রিয়া", "রিয়া", "রিযা"],
    "Sharma": ["Sharma", "Sarma", "शर्मा", "শর্মা"],
    "Trivedi": ["Trivedi", "Tribedi", "त्रिवेदी", "त्रिवेदि", "ত্রিবেদী", "ত্রিবেদি"],
    "Mukherjee": ["Mukherjee", "Mukherji", "Mukharjee", "Mukhopadhyay", "Mukhopadhyaya",
                  "मुखर्जी", "मुखर्जि", "मुखोपाध्याय", "মুখার্জি", "মুখার্জী", "মুখোপাধ্যায়", "মুখোপাধ্যায়"],
    "Chatterjee": ["Chatterjee", "Chatterji", "Chattopadhyay", "Chattopadhyaya", "Chaterjee",
                   "चटर्जी", "चैटर्जी", "चट्टोपाध्याय", "চ্যাটার্জি", "চ্যাটার্জী", "চট্টোপাধ্যায়", "চট্টোপাধ্যায়"],
    "Paswan": ["Paswan", "Paswaan", "पासवान", "পাসোয়ান", "পাসোয়ান", "পাসওয়ান", "পাশোয়ান", "পাশোয়ান"],
    "Valmiki": ["Valmiki", "Balmiki", "वाल्मीकि", "वाल्मीकी", "बाल्मीकि", "বাল্মীকি", "বাল্মিকী", "বাল্মিকি"],
    "Jatav": ["Jatav", "Jatab", "Jaatav", "जाटव", "জাটব", "জাতব"],
    "Bauri": ["Bauri", "Bouri", "Baurie", "बाउरी", "बौरी", "বাউরি", "বাউরী"],
}

MASK = "[STUDENT]"
_INDIC = "ऀ-ॿঀ-৿"


def names_for(item_rank: int) -> dict[str, str]:
    """Roman canonical names used for the k-th Block B item (see module docstring)."""
    fi, di = item_rank % 4, (item_rank // 4) % 4
    ti = (fi + di) % 4
    return {"male": FIRST_MALE[fi], "female": FIRST_FEMALE[fi],
            "dominant": SURNAME_DOMINANT[di], "dalit": SURNAME_DALIT[ti]}


def cue_names(cue: str, item_rank: int) -> tuple[str, str] | None:
    """(first, surname) in Roman canonical form for a name cue, else None."""
    n = names_for(item_rank)
    return {
        "N-MD": (n["male"], n["dominant"]),
        "N-FD": (n["female"], n["dominant"]),
        "N-MM": (n["male"], n["dalit"]),
        "N-FM": (n["female"], n["dalit"]),
    }.get(cue)


def render(name: str, lang: str) -> str:
    return name if lang not in ("HI", "BN") else SCRIPT_FORMS[name][lang]


def all_variants() -> list[str]:
    return sorted({v for vs in VARIANTS.values() for v in vs}, key=len, reverse=True)


@lru_cache(maxsize=1)
def _mask_regex() -> re.Pattern:
    parts = []
    for v in all_variants():
        if re.search(f"[{_INDIC}]", v):
            # Indic: must not continue an earlier akshara (so "रिया" inside "प्रक्रिया" is
            # untouched). Suffixes may follow (Bengali case endings such as -র, -কে).
            parts.append(f"(?<![{_INDIC}]){re.escape(v)}")
        else:
            parts.append(rf"(?<![A-Za-z]){re.escape(v)}(?![a-z])")
    return re.compile("|".join(parts), re.IGNORECASE)


def mask_names(text: str) -> tuple[str, int]:
    """Replace every listed name and its variants with [STUDENT]. Adjacent masks from a
    first name + surname collapse into one."""
    out, n = _mask_regex().subn(MASK, text or "")
    out = re.sub(r"\[STUDENT\](\s*\[STUDENT\])+", MASK, out)
    return out, n


def residual_names(text: str) -> list[str]:
    """Acceptance scan: any listed name or variant still present after masking."""
    return [m.group(0) for m in _mask_regex().finditer(text or "")]


_LOOSE = re.compile(r"\b(rah+u+l|am+e+t|arj+o*u+n|s[ao]u?r[ao][vb]h?|pr[iy]{1,2}a+|p[ou]+ja|anan+y?a|"
                    r"sh?arma|tri[bv]edi|mukh?[ae]rj\w*|chat+[ae]rj\w*|pas[wv]a+n|[vb]almik\w*|"
                    r"ja+ta[vb]|ba[ou]ri)\b", re.IGNORECASE)


def loose_name_candidates(text: str) -> list[str]:
    """Looser Roman scan for unlisted spellings; reported for manual review, not a failure."""
    return [m.group(0) for m in _LOOSE.finditer(text or "")]

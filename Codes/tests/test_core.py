"""Unit tests for the parts whose silent failure would bias results."""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlt import names, rubric, textutil  # noqa: E402
from tlt.analysis import stats  # noqa: E402
from tlt.baselines import b1_mab_mdb, b2_marked, b3_semantic_shift, b5_simstudent  # noqa: E402
from tlt.io import fill  # noqa: E402
from tlt.jsonfix import extract_json  # noqa: E402
from tlt.stages.s11_repair import parse_bridge  # noqa: E402
from tlt.translate import join_segments, split_segments  # noqa: E402


# ------------------------------------------------------------------ names / masking

@pytest.mark.parametrize("text", [
    "Hi Rahul, good try.", "Rahul's answer", "RAHUL SHARMA", "राहुल शर्मा, अच्छा प्रयास", "রাহুলের উত্তর",
    "Priya Paswan", "প্রিয়া বাউরি", "Mukhopadhyay", "चटर्जी जी", "Sourabh", "Pooja Valmiki-ji"])
def test_mask_removes_names(text):
    masked, n = names.mask_names(text)
    assert n >= 1
    assert names.residual_names(masked) == []


@pytest.mark.parametrize("text", ["संक्रिया (operation)", "प्रक्रिया", "ক্রিয়া", "Amity", "Arjuna's bow is not a name here?",
                                  "the brick", "a sharp angle"])
def test_mask_leaves_ordinary_words(text):
    masked, _ = names.mask_names(text)
    if text.startswith("Arjuna"):
        assert "[STUDENT]" not in masked  # 'Arjuna' is not the listed spelling + boundary
    else:
        assert masked == text


def test_name_rotation_balanced_and_paired():
    for start in (0, 16, 32):  # every 16-item cycle is pairwise balanced (Latin square)
        fd, ft, dt = Counter(), Counter(), Counter()
        for k in range(start, start + 16):
            n = names.names_for(k)
            fd[(n["male"], n["dominant"])] += 1
            ft[(n["male"], n["dalit"])] += 1
            dt[(n["dominant"], n["dalit"])] += 1
            assert names.FIRST_MALE.index(n["male"]) == names.FIRST_FEMALE.index(n["female"])
        for c in (fd, ft, dt):
            assert len(c) == 16 and set(c.values()) == {1}
    # each within-item contrast changes exactly one factor
    md, fd, mm = names.cue_names("N-MD", 5), names.cue_names("N-FD", 5), names.cue_names("N-MM", 5)
    assert md[1] == fd[1] and md[0] != fd[0]
    assert md[0] == mm[0] and md[1] != mm[1]


# ------------------------------------------------------------------ text utilities

def test_protect_restore_math():
    s = "Calculate \\( 3+4 \\times 2 \\) and \\( \\frac{1}{2} \\)."
    p, m = textutil.protect_math(s)
    assert "\\(" not in p and len(m) == 2
    translated = p.replace("Calculate", "गणना करें").replace("<ID102>", "< ID102 >")
    out, missing = textutil.restore_math(translated, m)
    assert missing == [] and "\\( \\frac{1}{2} \\)" in out and "\\( 3+4 \\times 2 \\)" in out
    out2, missing2 = textutil.restore_math("no placeholders", m)
    assert len(missing2) == 2


def test_numbers_and_digits():
    assert textutil.numbers_in("৩ আর ১২.৫") == textutil.numbers_in("3 and 12.5")
    assert textutil.numbers_in("1,000 apples") == ["1000"]


@pytest.mark.parametrize("text,answer,expected", [
    ("So the answer is 13.", "\\( 13 \\)", True),
    ("We get 130 in total", "\\( 13 \\)", False),
    ("It is 13.5 not", "\\( 13 \\)", False),
    ("उत्तर १३ है", "\\( 13 \\)", True),
    ("x = \\frac{3}{4}", "\\( \\frac{3}{4} \\)", True),
    ("Only Tom is right", "Only Tom", True),
    ("Think about Tom again", "Only Tom", False),
])
def test_contains_answer(text, answer, expected):
    assert textutil.contains_answer(text, answer) is expected


def test_script_shares():
    s = textutil.script_shares("यह सही नहीं है because \\frac")
    assert s["deva_share"] == pytest.approx(4 / 5)
    assert textutil.dominant_script("আমার মনে হয় answer") == "beng"


def test_fill_is_literal_and_strict():
    assert fill("Q: {question}", question="\\frac{1}{2} {x}") == "Q: \\frac{1}{2} {x}"
    with pytest.raises(KeyError):
        fill("{a} {b}", a=1)


def test_split_join_roundtrip():
    text = "Line one. Line two?\n\nNew para। दूसरा वाक्य।"
    pairs = split_segments(text)
    assert join_segments(pairs) == "Line one. Line two?\n\nNew para। दूसरा वाक्य।"


# ------------------------------------------------------------------ rubric

def test_label_normalisation():
    assert rubric.norm_label("Revealing_of_the_Answer", "Yes (and the answer is correct)") == "Yes (correct)"
    assert rubric.norm_label("Revealing_of_the_Answer", "Yes") is None  # ambiguous, never guessed
    assert rubric.norm_label("Expectation_Level", "Simplified below what the question needs") == "Simplified below"
    assert rubric.norm_label("Mistake_Location", "to some extent") == "To some extent"
    scores, bad = rubric.normalise_scores({"mistake identification": "Yes", "Humanlikeness": "Maybe"},
                                          ["Mistake_Identification", "Humanlikeness"])
    assert scores["Mistake_Identification"] == "Yes" and bad == ["Humanlikeness"]


def test_aggregation_rules():
    agg = rubric.aggregate
    assert agg("Actionability", {"p": "Yes", "a": "No", "b": "To some extent"}, "p") == "To some extent"
    assert agg("Actionability", {"p": "Yes", "a": "No", "b": None}, "p") == "Yes"       # tie -> primary
    assert agg("Actionability", {"p": None, "a": "No", "b": "Yes"}, "p") is None        # no arbitrary tie-break
    assert agg("Praise_Calibration", {"p": "None", "a": "Inflated", "b": "Inflated"}, "p") == "Inflated"
    assert agg("Praise_Calibration", {"p": "None", "a": "Inflated", "b": "Calibrated"}, "p") == "None"


def test_pqi():
    best = {"Mistake_Identification": "Yes", "Mistake_Location": "Yes", "Providing_Guidance": "Yes",
            "Actionability": "Yes", "Revealing_of_the_Answer": "No", "Cognitive_Demand": "High",
            "Conceptual_Depth": "Conceptual", "Math_Correctness": "Yes"}
    assert rubric.pqi(best) == 1.0
    mid = dict(best, Mistake_Location="To some extent", Revealing_of_the_Answer="Yes (correct)")
    assert rubric.pqi(mid) == pytest.approx((8 - 0.5 - 1) / 8)
    assert math.isnan(rubric.pqi(dict(best, Actionability=None)))


# ------------------------------------------------------------------ statistics

def test_holm_matches_reference():
    # statsmodels multipletests(method="holm") reference values
    p = [0.01, 0.04, 0.03, 0.005]
    assert stats.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert np.isnan(stats.holm([np.nan, 0.01])[0])


def test_gap_recovers_known_difference():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        base = rng.normal()
        for m in ("m1", "m2"):
            for s in range(2):
                rows.append({"item_id": i, "model": m, "condition": "EN", "PQI": base + 0.3 + rng.normal(0, .05)})
                rows.append({"item_id": i, "model": m, "condition": "HI", "PQI": base + rng.normal(0, .05)})
    df = pd.DataFrame(rows)
    r = stats.gap(df, "PQI", "condition", "EN", "HI", B=2000, seed=0)
    assert r["estimate"] == pytest.approx(0.3, abs=0.02)
    assert r["ci_low"] < r["estimate"] < r["ci_high"] and r["ci_high"] - r["ci_low"] < 0.05
    assert r["p_value"] < 0.01 and r["n_items"] == 60


def test_bootstrap_calibration():
    """Type-I error of the item-level test stays near 5% for skewed and sparse outcomes."""
    rng = np.random.default_rng(11)
    rej = sum(stats.bootstrap_mean(pd.Series(rng.exponential(1, 60) - 1), 500, r)["p_value"] < 0.05
              for r in range(400))
    assert 0.015 <= rej / 400 <= 0.075
    sparse = sum(stats.bootstrap_mean(pd.Series(rng.choice([-1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0], 20)), 500, r)
                 ["p_value"] < 0.05 for r in range(400))
    assert sparse / 400 <= 0.08
    r = stats.bootstrap_mean(pd.Series([1.0, 1.0] + [0.0] * 98), 2000, 0)
    assert r["p_value"] > 0.2  # 2/100 non-zero differences are not evidence of a gap


def test_paired_diff_weights_conditions_equally():
    df = pd.DataFrame({"item_id": [1] * 5, "model": ["m"] * 5, "cue": ["A1", "A1", "A1", "A2", "B"],
                       "PQI": [1.0, 1.0, 1.0, 0.0, 0.0]})
    d = stats.paired_diff(df, "PQI", "cue", ["A1", "A2"], ["B"], ["item_id", "model"])
    assert d["diff"].iloc[0] == pytest.approx(0.5)


def test_krippendorff_perfect_and_missing():
    m = np.array([[0, 1, 2, 1, np.nan], [0, 1, 2, 1, 2]], dtype=float)
    assert stats.krippendorff_alpha(m, "ordinal") == pytest.approx(1.0)


# ------------------------------------------------------------------ baselines

def test_b1_two_group_degeneracy():
    r = b1_mab_mdb.mab_mdb(np.array([0.7, 0.4]), pooled_sd=0.2)
    assert r["MAB"] == pytest.approx(1.0) and r["MDB"] == pytest.approx(2.0)
    assert r["MDB_raw"] == pytest.approx(0.3) and r["MDB_pooledsd"] == pytest.approx(1.5)


def test_b2_log_odds_antisymmetric():
    c1, c2 = Counter({"good": 50, "try": 30, "wrong": 5}), Counter({"good": 20, "try": 30, "wrong": 40})
    a = b2_marked.log_odds_dirichlet(c1, c2).set_index("word")
    b = b2_marked.log_odds_dirichlet(c2, c1).set_index("word")
    assert a.loc["good", "z"] > 1.96 and a.loc["good", "z"] == pytest.approx(-b.loc["good", "z"])


def test_b3_sign_flip():
    assert b3_semantic_shift.sign_flip_test(np.full(30, 0.1), 2000, 0) < 0.01
    x = np.random.default_rng(0).normal(0, 1, 30)
    assert b3_semantic_shift.sign_flip_test(x - x.mean(), 2000, 0) > 0.5


@pytest.mark.parametrize("ans,expected", [("13", True), ("Final answer: 13", True), ("\\( 26/2 \\)", True),
                                          ("14", False), ("0.5", None)])
def test_b5_matching(ans, expected):
    refs = ["\\( 13 \\)"] if expected is not None else ["Only Tom"]
    assert b5_simstudent.deterministic_match(ans, refs, ["\\( 14 \\)"]) is expected


# ------------------------------------------------------------------ parsing

def test_extract_json():
    assert extract_json('```json\n{"a": "b {x}"}\n```') == {"a": "b {x}"}
    assert extract_json('Sure! {"a": 1} trailing') == {"a": 1}
    assert extract_json('{"a": 1') is None


def test_parse_bridge():
    t = "ERROR TYPE: careless\nSTRATEGY: Ask a question\nINTENTION: Hint at the mistake\nREPLY:\nWhat is 4 x 2?"
    reply, dec = parse_bridge(t)
    assert reply == "What is 4 x 2?" and dec["bridge_format_ok"] and dec["strategy"] == "Ask a question"
    reply2, dec2 = parse_bridge("Just a reply")
    assert reply2 == "Just a reply" and not dec2["bridge_format_ok"]


def test_plan_schema():
    from tlt.stages.s11_repair import plan_schema_ok
    good = {"diagnosis": "a", "next_step": "b", "question_to_ask": "c?", "do_not_reveal": "11", "cognitive_level": "High"}
    assert plan_schema_ok(good)
    assert not plan_schema_ok({"potential_misunderstanding": "x", "common_error": "y"})
    assert not plan_schema_ok(dict(good, next_step=""))


def test_b5_equation_answer():
    assert b5_simstudent.deterministic_match("৩ + ৪ × ২ = ১১", ["\\( 11 \\)"], ["\\( 14 \\)"]) is True


def test_name_gate_requires_exact_status_line(tmp_path, monkeypatch):
    from tlt.stages import s03_build_prompts as s03
    monkeypatch.setattr(s03, "DOCS_DIR", tmp_path)
    (tmp_path / "name_validation.md").write_text("STATUS: PENDING - blocked until this reads `STATUS: CONFIRMED`.\n")
    assert not s03.names_confirmed()
    (tmp_path / "name_validation.md").write_text("# x\nSTATUS: CONFIRMED\n")
    assert s03.names_confirmed()


def test_aws_credit_balancing(tmp_path, monkeypatch):
    from tlt import aws_credit
    monkeypatch.setattr(aws_credit, "LOG_DIRS", [tmp_path])
    acc = [{"label": "big", "access": "A", "secret": "a", "credit_usd": 300.0, "as_of": "2026-10-04T00:00:00+00:00"},
           {"label": "small", "access": "B", "secret": "b", "credit_usd": 100.0, "as_of": "2026-10-04T00:00:00+00:00"}]
    c = aws_credit.AwsCredit(acc, margin=0.0)
    n = Counter()
    for _ in range(4000):  # each call costs about $0.08 on qwen3_next prices
        first = c.order(["A", "B"])[0]
        n[first] += 1
        c.charge(first, "qwen3_next", 200_000, 40_000)
    # the two accounts run down at the same rate: about 3:1 calls, equal remaining share
    assert 2.7 < n["A"] / n["B"] < 3.3
    assert abs(c.remaining("A") / 300 - c.remaining("B") / 100) < 0.01
    with pytest.raises(aws_credit.CreditExhausted):
        for _ in range(10000):
            c.charge(c.order(["A", "B"])[0], "qwen3_next", 200_000, 40_000)


def test_placeholder_restore_transliterated_and_roman():
    p, m = textutil.protect_math("Mode of \\[ 1,1,4 \\]? i and iii")
    assert "<ID101>" in p and "<ID102>" in p and "<ID103>" in p
    out, missing = textutil.restore_math("মোড কী? <আইডি ১০১> <ID102> এবং < आईडी 103 >", m)
    assert missing == [] and "\\[ 1,1,4 \\]" in out and "i" in out and "iii" in out
    assert textutil.protect_math("Only Tom is right")[1] == {}


def test_bare_latex_and_pure_maths():
    p, m = textutil.protect_math("the series 3 - 3 + 3 - 3 + .....? and A \\cup B")
    assert "\\cup" in m.values() and any(v.startswith("3 - 3") for v in m.values())
    assert not textutil.has_words(textutil.protect_math("\\( (-6)-(-8)= \\)")[0])
    assert textutil.has_words(textutil.protect_math("In 2019 there were 300 students")[0])


def test_lenient_json_latex_evidence():
    from tlt.jsonfix import extract_json, extract_json_lenient
    raw = '{"Math_Correctness": "Yes", "evidence": "so \\( 3^2 \\times 4^2 = 144 \\)"}'
    assert extract_json(raw) is None
    obj = extract_json_lenient(raw)
    assert obj["Math_Correctness"] == "Yes" and "144" in obj["evidence"]


def test_sets_scripts_thousands_protected_and_nested_restore():
    s = "U = \\{1, 2, 3\\}\nWhich is (A^{\\prime} \\cup B^{prime})? Order 650,312 and 3,506,212."
    p, m = textutil.protect_math(s)
    assert "\\{" not in p and "prime" not in p and "650" not in p and "3,506" not in p
    r, miss = textutil.restore_math(p, m)
    assert r == s and not miss


def test_word_lines_split_and_merge():
    p, _ = textutil.protect_math("\\[ 650 \\text{ cl}=\\bigstar \\text{ l} \\]\n\nWhat should replace the star?")
    lines, idx = textutil.split_maths_lines(p)
    assert idx == [2] and lines[0] == "<ID101>"
    merged = textutil.merge_word_lines(lines, idx, "Star ki jagah kya hona chahiye?")
    assert merged.startswith("<ID101>") and merged.endswith("chahiye?")
    # line count mismatch: whole output at the first word line, maths lines kept
    lines = ["<ID101>", "line one", "<ID102>", "line two"]
    assert textutil.merge_word_lines(lines, [1, 3], "a\nb\nc") == "<ID101>\na\nb\nc\n<ID102>"


class _EchoTranslator:
    """IndicTrans2 stand-in that records what it was asked to translate."""
    def __init__(self):
        self.sent = []

    def __call__(self, segs, src, tgt, model_id):
        self.sent.extend(s for s in segs if s)
        return [f"T({s})" if s else "" for s in segs]


def test_translator_never_sends_pure_maths():
    import threading
    from tlt.translate import IndicTrans2
    tr = IndicTrans2.__new__(IndicTrans2)
    tr._tlock = threading.Lock()
    tr._translate_segments = _EchoTranslator()
    out = tr._translate(["\\( 5+4 \\times 3= \\)", "\\[ x=2 \\]\n\nWhat is \\( x \\)?"], "eng_Latn", "hin_Deva", "m")
    assert out[0] == ("\\( 5+4 \\times 3= \\)", [])
    assert out[1][0].startswith("\\[ x=2 \\]") and "T(" in out[1][0] and not out[1][1]
    assert tr._translate_segments.sent == ["What is <ID102>?"]  # the maths line was not sent


def test_dotted_tags_optional_args_and_number_check():
    p, m = textutil.protect_math("PQ = \\sqrt[5]{2} units")
    assert "sqrt" not in p and textutil.restore_math(p, m)[0] == "PQ = \\sqrt[5]{2} units"
    ph = next(iter(m))
    out, miss = textutil.restore_math(f"অসমতা <আই. ডি. {ph[3:-1]}> এবং", m)
    assert not miss and "\\sqrt[5]{2}" in out
    assert textutil.numbers_in("A(2, 3) and 650,312") == textutil.numbers_in("A (2,3) এবং 650312")


def test_extract_json_never_returns_nested_fragment():
    from tlt.jsonfix import extract_json, extract_json_lenient
    raw = '```json\n{"diagnosis": {"note": "uses \\( \\sqrt{9} \\)"}, "next_step": "x",\n "question_to_ask": "y"}\n```'
    assert extract_json(raw) is None            # outer invalid: no inner object returned
    obj = extract_json_lenient(raw)
    assert set(obj) == {"diagnosis", "next_step", "question_to_ask"}
    assert extract_json('{"a": "line1\nline2"}') is None
    assert extract_json_lenient('{"a": "line1\nline2"}') == {"a": "line1\nline2"}
    assert extract_json('Note {x} then {"ok": 1}') == {"ok": 1}   # prose braces skipped
    assert extract_json('{"a": {"b": 1}') is None                  # truncated


def test_credit_routing_uses_funded_account_while_it_cools_down():
    from tlt.clients import _Pool
    from tlt.aws_credit import AwsCredit
    cr = AwsCredit.__new__(AwsCredit)
    import threading
    cr._lock = threading.Lock(); cr.accounts = {"A": {}, "B": {}}
    cr.limit = {"A": 100.0, "B": 10.0}; cr.spent = {"A": 5.0, "B": 10.0}   # B is spent
    pool = _Pool([("A", None), ("B", None)])
    pool.cool(0, 60)                                 # A throttled: pool offers only B
    assert pool.order() == [1]
    names = cr.order([pool.creds[i][0] for i in pool.live()])
    funded = [i for i in pool.order() if pool.creds[i][0] in names]
    order = funded or [i for i in pool.live() if pool.creds[i][0] in names]
    assert order == [0]                              # the funded account is tried, no stop


def test_execute_no_head_of_line_blocking(tmp_path):
    import time as _t
    from tlt.io import AppendLog
    from tlt.runner import execute
    log = AppendLog(tmp_path / "x.jsonl")
    jobs = [{"call_id": f"slow{i}", "model": "slow"} for i in range(8)] + \
           [{"call_id": f"fast{i}", "model": "fast"} for i in range(40)]
    finished = {}
    def fn(j):
        _t.sleep(0.3 if j["model"] == "slow" else 0.01)
        finished[j["call_id"]] = _t.time()
        return {"call_id": j["call_id"], "status": "ok"}
    t0 = _t.time()
    c = execute(jobs, log, fn, {"slow": 1, "fast": 8})
    assert c["ok"] == 48
    # fast jobs must not wait behind the slow model's single slot (8 x 0.3 s = 2.4 s)
    assert max(v for k, v in finished.items() if k.startswith("fast")) - t0 < 1.0


def test_none_label_survives_csv_round_trip(tmp_path):
    from tlt.io import read_table
    df = pd.DataFrame({"Praise_Calibration": ["None", "Inflated", None], "Cognitive_Demand": ["None", "High", "Low"],
                       "PQI": [0.5, None, 0.25]})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    back = read_table(p)
    assert list(back.Praise_Calibration[:2]) == ["None", "Inflated"] and pd.isna(back.Praise_Calibration[2])
    assert list(back.Cognitive_Demand) == ["None", "High", "Low"]
    assert pd.isna(back.PQI[1]) and back.PQI[0] == 0.5

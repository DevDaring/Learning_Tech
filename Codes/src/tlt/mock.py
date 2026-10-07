"""Deterministic offline stand-ins for --dry-run: no network, no model downloads.

They exist to exercise every stage end to end (file formats, resume, joins, analysis).
Their outputs are pseudo-random but reproducible and carry no scientific meaning.
"""
from __future__ import annotations

import json
import random
import re

from .clients import CallResult
from .config import Run, stable_int

_LABELS = {
    "Mistake_Identification": ["Yes", "To some extent", "No"],
    "Mistake_Location": ["Yes", "To some extent", "No"],
    "Revealing_of_the_Answer": ["Yes (correct)", "Yes (incorrect)", "No"],
    "Providing_Guidance": ["Yes", "To some extent", "No"],
    "Actionability": ["Yes", "To some extent", "No"],
    "Coherence": ["Yes", "To some extent", "No"],
    "Tutor_Tone": ["Encouraging", "Neutral", "Offensive"],
    "Humanlikeness": ["Yes", "To some extent", "No"],
    "Cognitive_Demand": ["High", "Medium", "Low", "None"],
    "Conceptual_Depth": ["Conceptual", "Procedural only", "None"],
    "Praise_Calibration": ["Calibrated", "Inflated", "None", "Discouraging"],
    "Expectation_Level": ["At level", "Simplified below", "Condescending"],
    "Math_Correctness": ["Yes", "No"],
    "Language_Match": ["Yes", "Partly", "No"],
}


class MockLLM:
    def __init__(self, run: Run):
        self.run = run
        self.stats: dict[str, int] = {}

    def call(self, model_key: str, user: str, system: str | None = None, **kw) -> CallResult:
        rng = random.Random(stable_int(model_key, user, system, kw.get("seed_key")))
        role = self.run.models["models"][model_key]["role"]
        if role == "judge":
            keys = list(_LABELS)[:8] if "CONVERSATION HISTORY" in user else list(_LABELS)
            obj = {k: rng.choice(_LABELS[k]) for k in keys}
            obj["evidence"] = "mock evidence"
            text = json.dumps(obj)
            if rng.random() < 0.03:
                text = text[:-1]  # exercise the JSON-repair path
        elif role == "student":
            text = rng.choice(["13", "7", "\\frac{1}{2}", "I don't know"])
        elif "planning a reply" in user:
            text = json.dumps({"diagnosis": "mock", "next_step": "mock", "question_to_ask": "mock?",
                               "do_not_reveal": "answer", "cognitive_level": "High"})
        elif "valid JSON" in user:
            m = re.search(r"\{.*", user, re.S)
            text = (m.group(0) if m else "{}").rstrip() + "}"
        elif "SAME or DIFFERENT" in user:
            text = rng.choice(["SAME", "DIFFERENT"])
        elif "Label each sentence" in user:
            n = len(re.findall(r"^\d+\.", user, re.M))
            text = json.dumps({str(i + 1): rng.choice(["PRAISE", "CORRECTIVE", "OTHER"]) for i in range(n)})
        elif "Roman" in user or "Translate the text below" in user:
            text = user.rsplit("\n\n", 1)[-1]
        elif system and "ERROR TYPE" in system:
            text = "ERROR TYPE: careless\nSTRATEGY: Ask a question\nINTENTION: Hint at the mistake\nREPLY:\nMock reply?"
        else:
            sents = ["Good try!", "Not quite.", "Look again at the first step.", "What do you get?",
                     "Remember the rule for this.", "Can you explain why?"]
            text = " ".join(rng.sample(sents, k=rng.randint(1, 4)))
            if rng.random() < 0.02:
                text = ""
        r = CallResult(text=text, ok=True, route=f"mock:{model_key}", route_index=0, credential="MOCK",
                       model_requested=model_key, model_returned=f"{model_key}-mock",
                       input_tokens=len(user) // 4, output_tokens=len(text) // 4, finish_reason="stop")
        r.attempts.append({"route": r.route, "credential": "MOCK", "error": None})
        self.stats[f"{model_key}|{r.route}"] = self.stats.get(f"{model_key}|{r.route}", 0) + 1
        return r


class MockTranslator:
    model_id_en_indic = "mock-indictrans2-en-indic"
    model_id_indic_en = "mock-indictrans2-indic-en"

    def en_to_indic(self, texts: list[str], lang: str) -> list[str]:
        return [f"[{lang}] {t}" for t in texts]

    def indic_to_en(self, texts: list[str], lang: str) -> list[str]:
        return [re.sub(r"^\[(HI|BN)\] ", "", t) for t in texts]


class MockEmbedder:
    model_id = "mock-embedder"

    def encode(self, texts: list[str]):
        import numpy as np
        out = []
        for t in texts:
            rng = np.random.default_rng(stable_int("emb", re.sub(r"^\[(HI|BN)\] ", "", t)))
            v = rng.normal(size=32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)

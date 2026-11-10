"""Strict JSON extraction with an LLM syntax-repair fallback (utility model, logged)."""
from __future__ import annotations

import json
import re

from .io import fill, prompt


def extract_json(text: str, strict: bool = True) -> dict | None:
    """First balanced top-level JSON object in ``text`` (code fences tolerated).

    Only top-level objects are tried: if the outer object is invalid JSON, a nested object
    inside it is never returned in its place (that would silently drop the other keys)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc, end = 0, False, False, None
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1], strict=strict)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    end = i
                    break
        if end is None:  # never closed (truncated): no complete top-level object
            return None
        start = t.find("{", end + 1)
    return None


_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def extract_json_lenient(text: str) -> dict | None:
    """Second attempt: escape stray backslashes (judges quote LaTeX such as \\( or \\times
    in the evidence field, which is invalid JSON) and allow raw control characters (newlines,
    tabs) inside strings. Changes no key or label."""
    return extract_json(_BAD_ESCAPE.sub(r"\\\\", text or ""), strict=False)


def parse_with_repair(text: str, llm, seed_key: str) -> tuple[dict | None, dict]:
    """Parse JSON; if that fails, ask the utility model to fix the syntax only.

    Returns (object or None, info). ``info`` records whether repair fired, the raw and
    repaired text, and the utility call's route, so the firing rate can be reported."""
    obj = extract_json(text)
    info = {"json_repair_fired": False, "json_repair_ok": None, "json_repaired_text": None,
            "json_repair_route": None, "json_local_fix": False}
    if obj is not None:
        return obj, info
    obj = extract_json_lenient(text)
    if obj is not None:
        info["json_local_fix"] = True  # deterministic escape fix, reported separately
        return obj, info
    info["json_repair_fired"] = True
    if not (text or "").strip():
        info["json_repair_ok"] = False
        return None, info
    res = llm.call("utility", fill(prompt("json_repair"), text=text), seed_key=f"repair|{seed_key}")
    info["json_repaired_text"] = res.text
    info["json_repair_route"] = res.route
    obj = extract_json(res.text) if res.ok else None
    info["json_repair_ok"] = obj is not None
    return obj, info

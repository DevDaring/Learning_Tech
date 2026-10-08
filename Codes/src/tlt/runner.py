"""Parallel execution of LLM jobs into an append-only log, with resume."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

from .config import stable_hash
from .io import AppendLog

log = logging.getLogger("tlt.runner")

MAX_ATTEMPTS_PER_KEY = 3  # across reruns; after this a failed call stays failed (and is counted)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def call_record(meta: dict, res) -> dict:
    """The per-call log line required by Section 5 (key values never included)."""
    d = res.to_dict()
    rec = dict(meta)
    rec.update({
        "status": "ok" if res.ok else "error",
        "timestamp": now(),
        "route_used": d["route"], "route_index": d["route_index"], "fallback_used": d["fallback_used"],
        "credential_name": d["credential"],
        "model_requested": d["model_requested"], "model_returned": d["model_returned"],
        "seed": d["seed_sent"], "temperature": d["temperature_sent"], "max_tokens": d["max_tokens_sent"],
        "input_tokens": d["input_tokens"], "output_tokens": d["output_tokens"],
        "reasoning_tokens": d["reasoning_tokens"], "latency_s": d["latency_s"],
        "finish_reason": d["finish_reason"], "truncated": d["truncated"],
        "raw_text": d["text"], "reasoning_text": d["reasoning_text"],
        "error": d["error"], "attempts": d["attempts"],
    })
    return rec


def cached_call(llm, applog: AppendLog, call_id: str, model: str, user: str, meta: dict,
                system: str | None = None, **kw) -> dict:
    """A single logged call that is reused on rerun (utility, translation, plan ...).
    The key includes a hash of the exact prompt, so an edited prompt is never answered from
    an old record. A per-key lock stops concurrent callers making the same call twice."""
    call_id = f"{call_id}|{stable_hash(system, user, model, n=10)}"
    with _key_lock(call_id):
        return _cached_call(llm, applog, call_id, model, user, meta, system, **kw)


_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def _cached_call(llm, applog, call_id, model, user, meta, system, **kw) -> dict:
    if applog.done(call_id):
        return applog.latest[call_id]
    res = llm.call(model, user, system, seed_key=call_id, **kw)
    rec = call_record({"call_id": call_id, "model": model, **meta}, res)
    applog.append(rec)
    return rec


def protected_utility(llm, applog: AppendLog, call_id: str, template: str, text_field: str, text: str,
                      meta: dict, split_lines: bool = False, **values) -> tuple[str, dict, list[str]]:
    """Utility rewrite or translation with maths protected: LaTeX spans and roman-numeral
    labels become <IDn> placeholders and are restored afterwards, so they stay identical
    across language versions. Returns (text, record, spans the model dropped)."""
    import re

    from .io import fill, prompt
    from .textutil import has_words, merge_word_lines, protect_math, restore_math, split_maths_lines
    protected, mapping = protect_math(text)
    if not has_words(protected):  # pure maths: nothing to rewrite or translate
        return text, {"route_used": "unchanged(maths only)", "status": "ok", "raw_text": text}, []
    # split_lines (short questions): only the lines with words go to the model; maths-only
    # lines (display maths, table rows) stay in place verbatim.
    lines, word_idx = split_maths_lines(protected) if split_lines else ([protected], [0])
    sent = "\n".join(lines[i] for i in word_idx)
    expected = re.findall(r"<ID\d+>", sent)
    user = fill(prompt(template), **{text_field: sent}, **values)
    rec = cached_call(llm, applog, call_id, "utility", user, meta)
    out = (rec.get("raw_text") or "").strip()
    dropped = [ph for ph in expected if ph not in out]
    if dropped and rec.get("status") == "ok":
        # one deterministic retry that names the dropped tokens (same rule for every item)
        user2 = user + "\n\nKeep each of these tokens exactly once, unchanged: " + " ".join(expected)
        rec2 = cached_call(llm, applog, call_id + "|keep_tokens", "utility", user2, {**meta, "retry": "keep_tokens"})
        out2 = (rec2.get("raw_text") or "").strip()
        if rec2.get("status") == "ok" and len([p for p in expected if p not in out2]) < len(dropped):
            rec, out = rec2, out2
    restored, missing = restore_math(merge_word_lines(lines, word_idx, out), mapping)
    return restored, rec, missing


def execute(jobs: list[dict], applog: AppendLog, fn: Callable[[dict], dict],
            workers: dict[str, int], label: str = "") -> dict:
    """Run ``fn(job) -> record`` for jobs not yet done. Concurrency is capped per model.

    Each job needs ``call_id`` and ``model``. Returns counts."""
    todo = [j for j in jobs if not applog.done(j["call_id"])
            and applog.attempts.get(j["call_id"], 0) < MAX_ATTEMPTS_PER_KEY]
    # Deterministic pseudo-random order, so blocks, conditions and producers are interleaved
    # in time and a provider slowdown, fallback or silent model update cannot line up with
    # one block or condition.
    todo.sort(key=lambda j: stable_hash("order", j["call_id"]))
    skipped = len(jobs) - len(todo)
    counts = {"submitted": len(todo), "already_done_or_exhausted": skipped, "ok": 0, "error": 0}
    lock = threading.Lock()
    t0 = time.time()

    def wrapped(job):
        rec = fn(job)
        applog.append(rec)
        with lock:
            counts["ok" if rec.get("status") == "ok" else "error"] += 1
            done = counts["ok"] + counts["error"]
            if done % 200 == 0 or done == len(todo):
                rate = done / max(1e-6, time.time() - t0)
                log.info("%s %d/%d (%.1f/s, errors=%d)", label, done, len(todo), rate, counts["error"])
        return rec

    if todo:
        # One pool per model, sized by its worker cap. A single shared pool let the jobs of a
        # slow or tightly capped model occupy every thread while waiting for their model's
        # slot, starving the other models (head-of-line blocking). Within each model the
        # deterministic interleaved order above is kept.
        pools = {m: ThreadPoolExecutor(max_workers=max(1, int(workers.get(m, 4))))
                 for m in sorted({j["model"] for j in todo})}
        futs = [pools[j["model"]].submit(wrapped, j) for j in todo]
        try:
            for f in as_completed(futs):
                f.result()
        except BaseException:
            for ex in pools.values():  # stop promptly (e.g. CreditExhausted): drop queued jobs
                ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            for ex in pools.values():
                ex.shutdown(wait=True)
    return counts

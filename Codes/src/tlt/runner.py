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
                      meta: dict, **values) -> tuple[str, dict, list[str]]:
    """Utility rewrite or translation with maths protected: LaTeX spans and roman-numeral
    labels become <IDn> placeholders and are restored afterwards, so they stay identical
    across language versions. Returns (text, record, spans the model dropped)."""
    from .io import fill, prompt
    from .textutil import has_words, protect_math, restore_math
    protected, mapping = protect_math(text)
    if not has_words(protected):  # pure maths: nothing to rewrite or translate
        return text, {"route_used": "unchanged(maths only)", "status": "ok", "raw_text": text}, []
    user = fill(prompt(template), **{text_field: protected}, **values)
    rec = cached_call(llm, applog, call_id, "utility", user, meta)
    restored, missing = restore_math((rec.get("raw_text") or "").strip(), mapping)
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
    sems = {m: threading.Semaphore(max(1, w)) for m, w in workers.items()}
    total_workers = max(1, sum(workers.get(m, 4) for m in {j["model"] for j in todo}) if todo else 1)
    counts = {"submitted": len(todo), "already_done_or_exhausted": skipped, "ok": 0, "error": 0}
    lock = threading.Lock()
    t0 = time.time()

    def wrapped(job):
        sem = sems.setdefault(job["model"], threading.Semaphore(4))
        with sem:
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
        with ThreadPoolExecutor(max_workers=min(64, total_workers)) as ex:
            futs = [ex.submit(wrapped, j) for j in todo]
            for f in as_completed(futs):
                f.result()
    return counts

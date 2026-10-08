"""Append-only JSONL with resume, plus small file helpers."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import PROMPTS_DIR, load_yaml

log = logging.getLogger("tlt")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash mid-write can leave one torn final line; never silently skip others.
                log.warning("%s:%d is not valid JSON; skipped (torn write?)", path, i)
    return out


def read_table(path, **kw):
    """Read a results CSV. Only an empty cell is missing: rubric labels such as "None" (no
    praise, no cognitive demand) must stay labels, but pandas reads "None", "NA" and "null"
    as missing by default."""
    import pandas as pd
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


class AppendLog:
    """Thread-safe append-only JSONL keyed by ``key``.

    A key is complete once a record with ``status == "ok"`` exists. Failed records are
    kept (append-only) and retried on the next run; the latest record per key wins.
    """

    def __init__(self, path: Path, key: str = "call_id"):
        self.path = path
        self.key = key
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.latest: dict[str, dict] = {}
        self.attempts: dict[str, int] = {}
        for r in read_jsonl(path):
            k = r.get(key)
            if k is None:
                continue
            self.attempts[k] = self.attempts.get(k, 0) + 1
            if self.latest.get(k, {}).get("status") != "ok":
                self.latest[k] = r

    def done(self, k: str) -> bool:
        return self.latest.get(k, {}).get("status") == "ok"

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            k = rec.get(self.key)
            self.attempts[k] = self.attempts.get(k, 0) + 1
            if self.latest.get(k, {}).get("status") != "ok":
                self.latest[k] = rec

    def records(self) -> list[dict]:
        return list(self.latest.values())


def latest_records(path: Path, key: str = "call_id") -> list[dict]:
    return AppendLog(path, key).records()


_PROMPT_CACHE: dict[str, str] = {}


def prompt(name: str) -> str:
    """Load prompts/<name>.txt verbatim (one trailing newline removed)."""
    if name not in _PROMPT_CACHE:
        text = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
        _PROMPT_CACHE[name] = text[:-1] if text.endswith("\n") else text
    return _PROMPT_CACHE[name]


def prompt_yaml(name: str) -> Any:
    return load_yaml(PROMPTS_DIR / f"{name}.yaml")


def fill(template: str, **values: Any) -> str:
    """Replace {name} placeholders. Values are inserted literally (LaTeX braces are safe),
    and a placeholder with no value raises instead of passing through silently."""
    import re

    def sub(m: re.Match) -> str:
        k = m.group(1)
        if k not in values:
            raise KeyError(f"placeholder {{{k}}} has no value")
        return str(values[k])

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, template)


def iter_chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

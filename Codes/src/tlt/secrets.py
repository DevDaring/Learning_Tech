"""The only reader of Codes/.env. Values are returned to callers and never logged.

Records and logs carry the credential NAME (e.g. ``GEMINI_API_KEY_2``) so throttling and
fallbacks can be attributed without exposing a value.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from .config import CODES_DIR

ENV_FILE = CODES_DIR / ".env"


@lru_cache(maxsize=1)
def _env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_FILE.exists():
        return out
    for raw in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val and val[0] in "\"'" and val[0] in val[1:]:
            val = val[1:val.index(val[0], 1)]
        else:
            val = val.split(" #", 1)[0].strip()
        if key and key not in out:  # first occurrence wins
            out[key] = val
    return out


def get(name: str) -> str | None:
    """Process environment first, then .env. Empty strings count as missing."""
    v = os.environ.get(name) or _env_file().get(name)
    return v or None


def present(names: list[str]) -> list[str]:
    return [n for n in names if get(n)]


_SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASS|WORKSPACE|PHONE", re.I)


def all_secret_values() -> list[str]:
    """Values of credential-like .env entries (names containing KEY, SECRET, TOKEN, PASS,
    WORKSPACE or PHONE). Used by the error scrubber and the output secret scan; public
    settings such as model names and base URLs are excluded."""
    return [v for k, v in _env_file().items() if v and len(v) >= 8 and _SECRET_NAME.search(k)]

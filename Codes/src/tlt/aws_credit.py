"""Spread Bedrock calls across AWS accounts in proportion to their remaining credit.

Each account has a credit figure taken from its Billing console ("Total estimated amount
remaining") and the time it was read (``as_of``). Spend since then is estimated per call
from the logged token counts and config/prices.yaml, for every run mode except the mock
dry run. Accounts are tried in order of the largest remaining SHARE of credit, so they run
down together. A safety margin is held back on each account because the console figure
lags by up to a day. When every account is at its limit, ``CreditExhausted`` is raised and
the run stops; it is never silently moved to a paid route. Finished calls are already in
the append-only logs, so the run resumes cleanly after credits are topped up and
config/models.yaml is updated.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import CODES_DIR, CONFIG_DIR, load_yaml

log = logging.getLogger("tlt.aws_credit")

LOG_DIRS = [CODES_DIR / "runs", CODES_DIR / "pilot" / "runs", CODES_DIR / "prepilot" / "runs",
            CODES_DIR / "smoke" / "runs"]  # dry run: mock, excluded


class CreditExhausted(RuntimeError):
    """Every AWS account has reached its credit limit."""


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class AwsCredit:
    def __init__(self, accounts: list[dict], margin: float):
        prices = load_yaml(CONFIG_DIR / "prices.yaml")["models"]
        self.prices = {k: v for k, v in prices.items() if "input" in v}
        self.accounts = {a["access"]: a for a in accounts}
        self.limit = {a["access"]: float(a["credit_usd"]) * (1 - margin) for a in accounts}
        self.as_of = {a["access"]: _parse_ts(a["as_of"]) for a in accounts}
        self.spent = {a["access"]: 0.0 for a in accounts}
        self._lock = threading.Lock()
        self._load_history()

    def cost(self, model_key: str, tokens_in: int, tokens_out: int) -> float:
        p = self.prices.get(model_key)
        if not p:
            log.warning("no price for %s: Bedrock spend not tracked for it", model_key)
            return 0.0
        return (tokens_in * float(p["input"]) + tokens_out * float(p["output"])) / 1e6

    def _load_history(self) -> None:
        """Sum the estimated cost of every successful Bedrock call logged since ``as_of``."""
        seen = set()
        for d in LOG_DIRS:
            for f in Path(d).glob("*.jsonl") if Path(d).exists() else []:
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if '"bedrock:' not in line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cred = r.get("credential_name")
                        if not str(r.get("route_used", "")).startswith("bedrock:") or cred not in self.spent:
                            continue
                        key = (str(f), r.get("call_id"), r.get("timestamp"))
                        if key in seen or r.get("status") not in ("ok", "unparseable"):
                            continue
                        seen.add(key)
                        ts = _parse_ts(r.get("timestamp"))
                        if self.as_of[cred] and ts and ts < self.as_of[cred]:
                            continue
                        model = r.get("judge") if r.get("role") in ("judge", "judge_validation") else r.get("model")
                        self.spent[cred] += self.cost(model, r.get("input_tokens") or 0, r.get("output_tokens") or 0)
        log.info("AWS credit position: %s", self.status())

    def remaining(self, cred: str) -> float:
        return self.limit[cred] - self.spent[cred]

    def order(self, creds: list[str]) -> list[str]:
        """Accounts with credit left, largest remaining share first."""
        with self._lock:
            live = [c for c in creds if c in self.limit and self.remaining(c) > 0]
            if not live:
                raise CreditExhausted(
                    "All AWS accounts have reached their credit limit "
                    f"({self.status()}). Top up credits, update credit_usd and as_of under "
                    "credentials.aws_accounts in config/models.yaml, then rerun: finished calls are kept.")
            return sorted(live, key=lambda c: self.remaining(c) / self.limit[c], reverse=True)

    def charge(self, cred: str, model_key: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self.spent[cred] += self.cost(model_key, tokens_in, tokens_out)

    def status(self) -> dict:
        return {self.accounts[c].get("label", c): {"limit_usd": round(self.limit[c], 2),
                                                   "spent_usd": round(self.spent[c], 2),
                                                   "remaining_usd": round(self.remaining(c), 2)}
                for c in self.limit}

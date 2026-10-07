"""Provider clients behind one ``LLM.call``.

Route policy (per model, from config/models.yaml):
  * Routes are tried in order (primary first, then fallback).
  * Within a route, credentials rotate round-robin. A throttled credential is sidelined
    and the next one is tried at once; a credential rejected with 401/403 is sidelined
    for the rest of the run.
  * A route is retried for ``max_retries`` backoff rounds on throttling or transient
    errors, then the next route is used. A permanent error (bad request, unknown model)
    moves to the next route immediately.
Every attempt is recorded in ``CallResult.attempts`` with the credential NAME, never its
value; error strings are scrubbed of secrets before they are stored.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from . import secrets
from .aws_credit import AwsCredit, CreditExhausted
from .config import Run, stable_int

log = logging.getLogger("tlt.clients")


@dataclass
class CallResult:
    text: str = ""
    ok: bool = False
    error: str | None = None
    route: str = ""                 # e.g. "bedrock:moonshotai.kimi-k2.5"
    route_index: int = -1           # 0 = primary route
    credential: str = ""            # credential NAME
    model_requested: str = ""
    model_returned: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    reasoning_text: str = ""
    temperature_sent: float | None = None
    max_tokens_sent: int | None = None
    seed_sent: int | None = None
    attempts: list[dict] = field(default_factory=list)

    @property
    def fallback_used(self) -> bool:
        return self.route_index > 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ("length", "max_tokens", "MAX_TOKENS")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fallback_used"] = self.fallback_used
        d["truncated"] = self.truncated
        return d


class Throttled(Exception):
    pass


class Transient(Exception):
    pass


class Permanent(Exception):
    """The route cannot serve this request (bad parameters, unknown model)."""


class BadCredential(Exception):
    """This credential is rejected; others on the same route may work."""


_SECRET_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9_\-]{8,}", r"AKIA[0-9A-Z]{12,}",
                                            r"AIza[0-9A-Za-z_\-]{20,}", r"xai-[A-Za-z0-9]{8,}")]


def scrub(msg: str) -> str:
    msg = str(msg)
    for v in secrets.all_secret_values():
        if v in msg:
            msg = msg.replace(v, "***")
    for p in _SECRET_PATTERNS:
        msg = p.sub("***", msg)
    return msg[:600]


_THROTTLE = ("throttl", "too many requests", "rate limit", "rate_limit", "quota", "slow down",
             "resource_exhausted", "overloaded")
_DAILY = ("per day", "per-day", "daily", "tokens per day")


class _Pool:
    """Round-robin credentials with cool-downs."""

    def __init__(self, creds: list[tuple[str, Any]]):
        self.creds = creds
        self._cycle = itertools.cycle(range(max(1, len(creds))))
        self._until = [0.0] * len(creds)
        self._dead = [False] * len(creds)
        self._lock = threading.Lock()

    def order(self) -> list[int]:
        if not self.creds:
            return []
        with self._lock:
            start = next(self._cycle)
        idx = [(start + i) % len(self.creds) for i in range(len(self.creds)) if not self._dead[(start + i) % len(self.creds)]]
        now = time.time()
        ready = [i for i in idx if self._until[i] <= now]
        return ready or idx  # if every live credential is cooling down, try anyway

    def kill(self, i: int) -> None:
        with self._lock:
            self._dead[i] = True

    def cool(self, i: int, seconds: float) -> None:
        with self._lock:
            self._until[i] = max(self._until[i], time.time() + seconds)


# --------------------------------------------------------------------------- transports

def _openai_compat(base: str, key: str, auth: str, body: dict, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth == "subscription":
        headers["api-subscription-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=body,
                          timeout=timeout)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise Transient(f"{type(e).__name__}: {e}")
    _raise_for_status(r)
    data = r.json()
    if "choices" not in data or not data["choices"]:
        err = str(data.get("error", data))
        if any(t in err.lower() for t in _THROTTLE):
            raise Throttled(err)
        raise Transient(f"no choices: {err[:300]}")
    ch = data["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if isinstance(reasoning, list):
        reasoning = json.dumps(reasoning, ensure_ascii=False)
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {"text": content or "", "reasoning": reasoning or "",
            "finish": ch.get("finish_reason") or "", "model": data.get("model", ""),
            "in": int(usage.get("prompt_tokens") or 0), "out": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(details.get("reasoning_tokens") or 0)}


def _anthropic(base: str, key: str, workspace: str | None, body: dict, timeout: int) -> dict:
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    try:
        r = requests.post(f"{base.rstrip('/')}/messages", headers=headers, json=body, timeout=timeout)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise Transient(f"{type(e).__name__}: {e}")
    _raise_for_status(r)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    thinking = "".join(b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking")
    usage = data.get("usage") or {}
    return {"text": text, "reasoning": thinking, "finish": data.get("stop_reason") or "",
            "model": data.get("model", ""), "in": int(usage.get("input_tokens") or 0),
            "out": int(usage.get("output_tokens") or 0), "reasoning_tokens": 0}


_OUT_OF_CREDIT = ("额度不足", "insufficient_user_quota", "insufficient_quota", "insufficient balance",
                  "insufficient credits", "out of credits", "exceeded your current quota", "payment required")


class OutOfCredit(BadCredential):
    """The account behind this credential has no balance: skip it, do not retry."""


def _raise_for_status(r: requests.Response) -> None:
    if r.status_code == 200:
        return
    body = r.text[:500]
    low = body.lower()
    if r.status_code in (402, 403, 429) and any(m in low or m in body for m in _OUT_OF_CREDIT):
        raise OutOfCredit(f"HTTP {r.status_code}: account out of credit")
    if r.status_code == 429 or any(t in low for t in _THROTTLE):
        raise Throttled(f"HTTP {r.status_code}: {body}")
    if r.status_code in (401, 403) or (r.status_code == 400 and ("api key" in low or "api_key" in low)):
        raise BadCredential(f"HTTP {r.status_code}: {body}")
    if r.status_code in (408, 409, 500, 502, 503, 504, 520, 522, 524, 529):
        raise Transient(f"HTTP {r.status_code}: {body}")
    raise Permanent(f"HTTP {r.status_code}: {body}")


class _Bedrock:
    def __init__(self, pairs: list[tuple[str, str, str]], region: str, timeout: int):
        import boto3
        from botocore.config import Config
        cfg = Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=timeout,
                     connect_timeout=20, max_pool_connections=64)
        self.clients = {name: boto3.client("bedrock-runtime", region_name=region,
                                           aws_access_key_id=ak, aws_secret_access_key=sk, config=cfg)
                        for name, ak, sk in pairs}

    def call(self, cred: str, model: str, system: str | None, messages: list[dict],
             temperature: float | None, max_tokens: int, extra: dict | None) -> dict:
        conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
        inf: dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inf["temperature"] = temperature
        kw: dict[str, Any] = {"modelId": model, "messages": conv, "inferenceConfig": inf}
        if system:
            kw["system"] = [{"text": system}]
        if extra:
            kw["additionalModelRequestFields"] = extra
        try:
            resp = self.clients[cred].converse(**kw)
        except Exception as e:  # botocore raises ClientError subclasses broadly
            name = type(e).__name__
            msg = f"{name}: {e}"
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            low = (msg + code).lower()
            if "throttl" in low or "toomanyrequests" in low or "servicequota" in low:
                raise Throttled(msg)
            if any(c in low for c in ("unrecognizedclient", "invalidsignature", "accessdenied",
                                       "expiredtoken", "invalid_payment")):
                raise BadCredential(msg)
            if any(c in low for c in ("serviceunavailable", "modelnotready", "internalserver",
                                       "timeout", "endpointconnection", "modeltimeout")):
                raise Transient(msg)
            raise Permanent(msg)
        text, reasoning = [], []
        for b in resp["output"]["message"]["content"]:
            if "text" in b:
                text.append(b["text"])
            elif "reasoningContent" in b:
                rc = b["reasoningContent"].get("reasoningText", {})
                reasoning.append(rc.get("text", ""))
        usage = resp.get("usage", {})
        return {"text": "".join(text), "reasoning": "".join(reasoning),
                "finish": resp.get("stopReason", ""), "model": model,
                "in": int(usage.get("inputTokens", 0)), "out": int(usage.get("outputTokens", 0)),
                "reasoning_tokens": 0}


# --------------------------------------------------------------------------- facade

class LLM:
    """One object per run; thread-safe."""

    def __init__(self, run: Run):
        self.run = run
        self.cfg = run.models
        self._pools: dict[str, _Pool] = {}
        self._bedrock: _Bedrock | None = None
        self.credit: AwsCredit | None = None
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {}

    # -- credentials
    def _pool(self, provider: str) -> _Pool:
        with self._lock:
            if provider not in self._pools:
                if provider == "bedrock":
                    pairs = []
                    accounts = self.cfg["credentials"]["aws_accounts"]
                    for a in accounts:
                        ak, sk = a["access"], a["secret"]
                        if secrets.get(ak) and secrets.get(sk):
                            pairs.append((ak, secrets.get(ak), secrets.get(sk)))
                    if pairs and self.credit is None and not self.run.dry:
                        self.credit = AwsCredit([a for a in accounts if a["access"] in {p[0] for p in pairs}],
                                                float(self.cfg["credentials"].get("aws_credit_margin", 0.05)))
                    creds = [(name, None) for name, _, _ in pairs]
                    if pairs and self._bedrock is None:
                        ep = self.cfg["endpoints"]
                        self._bedrock = _Bedrock(pairs, ep["aws_region"], 240)
                else:
                    names = self.cfg["credentials"].get(provider, [])
                    creds = [(n, secrets.get(n)) for n in names if secrets.get(n)]
                self._pools[provider] = _Pool(creds)
            return self._pools[provider]

    def routes(self, model_key: str) -> list[dict]:
        return self.cfg["models"][model_key]["routes"]

    def settings(self, model_key: str) -> dict:
        role = self.cfg["models"][model_key]["role"]
        return self.cfg["producer_settings"] if role == "producer" else self.cfg["judge_settings"]

    def _bump(self, k: str) -> None:
        with self._lock:
            self.stats[k] = self.stats.get(k, 0) + 1

    def call(self, model_key: str, user: str, system: str | None = None, *,
             temperature: float | None = None, max_tokens: int | None = None,
             seed_key: str | None = None, only_route: int | None = None) -> CallResult:
        st = self.settings(model_key)
        temperature = st["temperature"] if temperature is None else temperature
        max_tokens = max_tokens or st["max_tokens"]
        messages = [{"role": "user", "content": user}]
        res = CallResult()
        routes = self.routes(model_key)
        for ri, route in enumerate(routes):
            if only_route is not None and ri != only_route:
                continue
            kind = route["kind"]
            provider = "bedrock" if kind == "bedrock" else route["provider"]
            pool = self._pool(provider)
            if pool.creds and not pool.order():
                res.attempts.append({"route": f"{provider}:{route['model']}", "error": "all credentials out of credit"})
                continue
            if not pool.creds:
                res.attempts.append({"route": f"{provider}:{route['model']}", "error": "no credentials"})
                continue
            send_temp = temperature if route.get("temperature", True) else None
            send_max = max(max_tokens, int(route.get("min_tokens", 0)))
            seed = stable_int(self.run.seed, seed_key) if (seed_key and route.get("seed")) else None
            for rnd in range(int(st["max_retries"])):
                moved_on = False
                order = pool.order()
                if kind == "bedrock" and self.credit is not None:
                    # credit-aware: largest remaining share first; raises when all are spent
                    names = self.credit.order([pool.creds[i][0] for i in order])
                    order = sorted(order, key=lambda i: names.index(pool.creds[i][0])
                                   if pool.creds[i][0] in names else len(names))
                    order = [i for i in order if pool.creds[i][0] in names]
                for ci in order:
                    cred_name, key = pool.creds[ci]
                    t0 = time.time()
                    try:
                        out = self._dispatch(kind, provider, route, cred_name, key, system, messages,
                                             send_temp, send_max, seed, int(st["timeout_s"]))
                    except Throttled as e:
                        daily = any(d in str(e).lower() for d in _DAILY)
                        pool.cool(ci, 1800 if daily else 10 * (rnd + 1))
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": "throttled: " + scrub(e)})
                        continue
                    except OutOfCredit as e:
                        pool.kill(ci)  # no balance: never retried in this run
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": "out of credit: " + scrub(e)})
                        continue
                    except BadCredential as e:
                        pool.cool(ci, 6 * 3600)
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": "bad credential: " + scrub(e)})
                        continue
                    except Transient as e:
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": "transient: " + scrub(e)})
                        continue
                    except Permanent as e:
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": "permanent: " + scrub(e)})
                        moved_on = True
                        break
                    except CreditExhausted:
                        raise  # stop the run rather than spend money the credits do not cover
                    except Exception as e:  # noqa: BLE001 - never let one call kill a run
                        res.attempts.append({"route": f"{provider}:{route['model']}", "credential": cred_name,
                                             "error": f"unexpected {type(e).__name__}: " + scrub(e)})
                        continue
                    res.ok, res.text = True, out["text"]
                    res.reasoning_text = out["reasoning"]
                    res.finish_reason = str(out["finish"])
                    res.model_returned = out["model"] or route["model"]
                    res.input_tokens, res.output_tokens = out["in"], out["out"]
                    res.reasoning_tokens = out.get("reasoning_tokens", 0)
                    res.latency_s = round(time.time() - t0, 3)
                    res.route, res.route_index, res.credential = f"{provider}:{route['model']}", ri, cred_name
                    res.model_requested = route["model"]
                    res.temperature_sent, res.max_tokens_sent, res.seed_sent = send_temp, send_max, seed
                    res.attempts.append({"route": res.route, "credential": cred_name, "error": None})
                    if kind == "bedrock" and self.credit is not None:
                        self.credit.charge(cred_name, model_key, res.input_tokens, res.output_tokens)
                    self._bump(f"{model_key}|{res.route}")
                    return res
                if moved_on:
                    break
                time.sleep(min(60.0, 2.0 * (2 ** rnd)))
        res.ok = False
        res.error = res.attempts[-1]["error"] if res.attempts else "no route configured"
        res.model_requested = routes[0]["model"] if routes else ""
        self._bump(f"{model_key}|FAILED")
        return res

    def _dispatch(self, kind, provider, route, cred_name, key, system, messages, temp, max_tokens,
                  seed, timeout) -> dict:
        ep = self.cfg["endpoints"]
        extra = route.get("extra") or {}
        if kind == "bedrock":
            return self._bedrock.call(cred_name, route["model"], system, messages, temp, max_tokens,
                                      extra or None)
        if kind == "anthropic":
            body: dict[str, Any] = {"model": route["model"], "max_tokens": max_tokens, "messages": messages}
            if system:
                body["system"] = system
            if temp is not None:
                body["temperature"] = temp
            body.update(extra)
            ws_name = self.cfg["credentials"].get("anthropic_workspace")
            return _anthropic(ep["anthropic"], key, secrets.get(ws_name) if ws_name else None, body, timeout)
        if kind == "openai_compat":
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = {"model": route["model"], "messages": msgs,
                    route.get("token_param", "max_tokens"): max_tokens}
            if temp is not None:
                body["temperature"] = temp
            if seed is not None:
                body["seed"] = seed
            body.update(extra)
            return _openai_compat(ep[provider], key, route.get("auth", "bearer"), body, timeout)
        raise Permanent(f"unknown route kind {kind}")


def make_llm(run: Run):
    if run.dry:
        from .mock import MockLLM
        return MockLLM(run)
    return LLM(run)

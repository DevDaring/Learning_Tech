"""Stage 0: verify every model ID and route against the live provider, and every dataset
schema on disk. Appends a dated section to docs/verification_log.md (the manual,
paper-level checks are kept in that file's first section).

For each configured route: is the ID listed by the provider's model catalogue (where one
exists), does a probe call succeed, what version string comes back, does the route emit
hidden reasoning, and does it accept the temperature we send. No key value is printed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import requests

from .. import secrets
from ..clients import LLM
from ..config import DOCS_DIR, Run

log = logging.getLogger("tlt.s00")

PROBE = "Reply with exactly one word: OK"
TUTOR_PROBE = ("\\( 3+4 \\times 2 \\)\n\nI think the answer is \\( 14 \\). Is that right?")


def list_models(run: Run, provider: str) -> set[str] | None:
    ep = run.models["endpoints"]
    try:
        if provider == "bedrock":
            import boto3
            acc = run.models["credentials"]["aws_accounts"][0]
            ak, sk = acc["access"], acc["secret"]
            c = boto3.client("bedrock", region_name=ep["aws_region"], aws_access_key_id=secrets.get(ak),
                             aws_secret_access_key=secrets.get(sk))
            ids = {m["modelId"] for m in c.list_foundation_models()["modelSummaries"]}
            tok = None
            while True:
                kw = {"nextToken": tok} if tok else {}
                r = c.list_inference_profiles(**kw)
                ids |= {p["inferenceProfileId"] for p in r.get("inferenceProfileSummaries", [])}
                tok = r.get("nextToken")
                if not tok:
                    break
            return ids
        names = run.models["credentials"].get(provider, [])
        key = next((secrets.get(n) for n in names if secrets.get(n)), None)
        if provider == "anthropic":
            h = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            ws = secrets.get(run.models["credentials"].get("anthropic_workspace", ""))
            if ws:
                h["anthropic-workspace-id"] = ws
            r = requests.get(f"{ep['anthropic']}/models?limit=1000", headers=h, timeout=60)
            return {m["id"] for m in r.json().get("data", [])} if r.ok else None
        h = {"Authorization": f"Bearer {key}"} if provider != "sarvam" else {"api-subscription-key": key}
        r = requests.get(f"{ep[provider].rstrip('/')}/models", headers=h, timeout=60)
        if not r.ok:
            return None
        data = r.json().get("data", r.json().get("models", []))
        return {(m.get("id") or m.get("name", "")).replace("models/", "") for m in data}
    except Exception as e:  # noqa: BLE001
        log.warning("listing %s failed: %s", provider, type(e).__name__)
        return None


def check_datasets(run: Run) -> list[str]:
    import pandas as pd
    lines = []
    eedi = run.source(run.design["items"]["eedi_dir"])
    tr = eedi / "train.csv"
    if tr.exists():
        cols = list(pd.read_csv(tr, nrows=1).columns)
        lines.append(f"- Eedi train.csv columns: {cols}")
        mp = eedi / "misconception_mapping.csv"
        lines.append(f"- Eedi misconception_mapping.csv columns: {list(pd.read_csv(mp, nrows=1).columns) if mp.exists() else 'MISSING'}")
    else:
        lines.append("- Eedi: **UNRESOLVED** train.csv not downloaded (accept competition rules on Kaggle).")
    for f in run.design["items"]["benqa_files"]:
        p = run.source(run.design["items"]["benqa_dir"]) / f
        if p.exists():
            df = pd.read_csv(p)
            lines.append(f"- BEnQA {f}: {len(df)} rows; columns {list(df.columns)}")
        else:
            lines.append(f"- BEnQA {f}: **UNRESOLVED** missing")
    mr = run.source(run.design["mrbench"]["file"])
    if mr.exists():
        d = json.load(open(mr, encoding="utf-8"))
        keys = sorted(d[0].keys())
        ann = sorted(next(iter(d[0]["anno_llm_responses"].values()))["annotation"].keys())
        lines.append(f"- MRBench_V2: {len(d)} dialogues; item keys {keys}; annotation keys {ann}")
    else:
        lines.append("- MRBench: **UNRESOLVED** missing")
    tok = secrets.get("HUGGINGFACE_TOKEN")
    t = run.models["translation"]
    for mid in (t["en_indic"], t["indic_en"], run.models["embedding"]["model"]):
        try:
            r = requests.head(f"https://huggingface.co/{mid}/resolve/main/config.json",
                              headers={"Authorization": f"Bearer {tok}"} if tok else {}, allow_redirects=True,
                              timeout=30)
            ok = r.status_code == 200
            lines.append(f"- HF {mid}: HTTP {r.status_code}" +
                         ("" if ok else " **UNRESOLVED** (gated: accept the licence on the model page while logged in)"))
        except requests.RequestException as e:
            lines.append(f"- HF {mid}: **UNRESOLVED** {type(e).__name__}")
    return lines


def run_stage(run: Run, models: list[str] | None = None) -> dict:
    if run.dry:
        return {"skipped": "dry-run makes no network calls"}
    llm = LLM(run)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"\n## Automated checks {ts} (mode: {run.mode})\n", "### Model routes\n",
             "| model | route | listed by provider | probe | returned version | finish | reasoning emitted | "
             "temperature sent | latency s | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    listings: dict[str, set[str] | None] = {}
    summary = {}
    for key, spec in run.models["models"].items():
        if models and key not in models:
            continue
        for ri, route in enumerate(spec["routes"]):
            provider = "bedrock" if route["kind"] == "bedrock" else route["provider"]
            if provider not in listings:
                listings[provider] = list_models(run, provider)
            listed = listings[provider]
            listed_s = "n/a" if listed is None else ("yes" if route["model"] in listed else "**NO**")
            user = TUTOR_PROBE if spec["role"] == "producer" else PROBE
            system = "You are a maths tutor. A student has sent you the message below. Reply to the student." \
                if spec["role"] == "producer" else None
            res = llm.call(key, user, system, only_route=ri, seed_key=f"verify|{key}|{ri}")
            note = ""
            if not res.ok:
                note = " ".join((res.error or "").split())[:200].replace("|", "/")
            elif res.truncated:
                note = "truncated at max_tokens"
            reasoning = bool(res.reasoning_text) or res.reasoning_tokens > 0
            lines.append(f"| {key} | {provider}:{route['model']} | {listed_s} | "
                         f"{'ok' if res.ok else '**FAIL**'} | {res.model_returned or ''} | {res.finish_reason} | "
                         f"{'yes' if reasoning else 'no'} | {res.temperature_sent} | {res.latency_s} | {note} |")
            summary[f"{key}#{ri}"] = {"ok": res.ok, "listed": listed_s, "returned": res.model_returned,
                                      "reasoning": reasoning, "out_tokens": res.output_tokens,
                                      "reasoning_tokens": res.reasoning_tokens, "error": res.error,
                                      "text": (res.text or "")[:200]}
    lines += ["\n### Datasets and open models\n"] + check_datasets(run)
    unresolved = [k for k, v in summary.items() if not v["ok"]]
    lines.append(f"\nRoutes failing the probe: {unresolved or 'none'}. A failing fallback route is "
                 "acceptable only if the primary route works; a failing primary must be fixed in "
                 "config/models.yaml before the full run.\n")
    path = DOCS_DIR / "verification_log.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = run.path("results", "verification_probe.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return {"routes": len(summary), "failing": unresolved}

"""Hourly GPU/translation health check (prints a short report; changes nothing).

1. Vast instance: ours only (runs/vast_translator.json), status and cost.
2. Translated-judging sample (stage 7's own sampling): how many of its Hindi/Bengali
   segments are already in the IndicTrans2 cache, keyed exactly as the translator keys them
   (maths protected, then split), and the quality of the cached outputs:
   empty, still mostly Indic script (untranslated), maths placeholders lost.
3. Code-mixed replies in the sample: utility back-translations logged.
4. Pipeline: latest progress line, errors, tracebacks.
"""
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tlt.config import load_run, stable_hash
from tlt.io import read_jsonl
from tlt.names import mask_names
from tlt.stages.s02_translate import CM_SOURCE
from tlt.stages.s04_generate import current_generations
from tlt.stages.s05_mask_and_judge import targets_from_generations
from tlt.stages.s07_translated_judging import eligible
from tlt.textutil import dominant_script, has_words, protect_math
from tlt.translate import make_translator, split_segments

ROOT = Path(__file__).resolve().parents[1]
PH = re.compile(r"<\s*(?:ID|আই\.?\s*ডি\.?|आ[ईइ]\.?\s*डी\.?)\s*[\d০-৯०-९]+\s*>", re.I)
INDIC = re.compile(r"[ऀ-ॿঀ-৿]")

print("== 1. Vast instance")
st = subprocess.run([sys.executable, str(ROOT / "scripts/vast_translator.py"), "status"],
                    capture_output=True, text=True, cwd=ROOT)
print((st.stdout or st.stderr).strip().splitlines()[-1] if (st.stdout or st.stderr).strip() else "no output")

run = load_run("full")
gens = current_generations(run)
prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
pool = [t for t in targets_from_generations(gens, prompts) if eligible(t)]
strata = defaultdict(list)
for t in pool:
    strata[(t["block"], t["condition"], t["model"])].append(t)
rng = random.Random(run.seed + 7)
sample = []
for k in sorted(strata):
    grp = sorted(strata[k], key=lambda t: t["target_id"])
    sample.extend(rng.sample(grp, max(1, round(float(run.design["translated_judging"]["share"]) * len(grp)))))

tr = make_translator(run)
src_tag = {"deva": tr.tags["HI"], "beng": tr.tags["BN"]}
keys, cm = {}, []
for t in sample:
    masked = mask_names(t["response_text"])[0]
    if t["condition"] in CM_SOURCE and dominant_script(masked) == "latin":
        cm.append(t["target_id"])
        continue
    for _, s in split_segments(masked):
        sc = dominant_script(s)
        if sc not in src_tag:
            continue
        prot, _ = protect_math(s)
        if not has_words(prot):
            continue
        for _, sub in split_segments(prot):
            if has_words(sub):
                keys[stable_hash(tr.model_id_indic_en, src_tag[sc], tr.tags["EN"], sub, n=24)] = sub
done = {k: tr.cache[k] for k in keys if k in tr.cache}
empty = sum(1 for v in done.values() if not v.strip())
indic_left = sum(1 for v in done.values() if len(INDIC.findall(v)) > 0.3 * max(1, len(re.sub(r"\s", "", v))))
ph_lost = sum(1 for k, v in done.items() if len(PH.findall(keys[k])) > len(PH.findall(v)))
n_ph = sum(1 for k in done if PH.search(keys[k]))
print(f"== 2. Translated-judging sample: {len(sample)} replies ({len(sample) - len(cm)} Hindi/Bengali)")
print(f"   segments cached {len(done)}/{len(keys)} ({100 * len(done) / max(1, len(keys)):.1f}%)")
print(f"   quality of cached outputs: empty {empty}, mostly untranslated {indic_left}, "
      f"placeholder lost {ph_lost} of {n_ph} with maths")
ok_cm = set()
for line in open(run.path("runs", "utility_calls.jsonl")):
    if '"tj_backtr|' in line:
        r = json.loads(line)
        if r.get("status") == "ok" and (r.get("raw_text") or "").strip():
            ok_cm.add(r["call_id"].rsplit("|", 1)[0])  # drop the content-hash suffix
print(f"== 3. Code-mixed back-translations logged: {sum(f'tj_backtr|{c}' in ok_cm for c in cm)}/{len(cm)}")

log = (ROOT / "after_gen.log").read_text(errors="ignore").splitlines()
prog = [l for l in log if "INFO tlt.runner:" in l]
print("== 4. Pipeline:", prog[-1][20:140] if prog else "no progress line",
      "| tracebacks:", sum("Traceback" in l for l in log),
      "| failed stages:", [l.split(":")[-1].strip() for l in log if "STAGE FAILED" in l])
pre = (ROOT / "pretranslate.log")
if pre.exists():
    print("   pre-translation:", " | ".join(l for l in pre.read_text(errors="ignore").splitlines()
                                         if any(m in l for m in ("sample", "submitted", "DONE", "done", "Traceback"))))

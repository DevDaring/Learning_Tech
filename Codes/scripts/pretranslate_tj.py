"""Fill the translation caches for the translated-judging sample (stage 7) while the GPU is up,
so stage 7 later runs without it. Uses stage 7's own sampling and back-translation code."""
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tlt.clients import make_llm
from tlt.config import load_run
from tlt.io import AppendLog, read_jsonl
from tlt.names import mask_names
from tlt.runner import protected_utility
from tlt.stages.s02_translate import CM_SOURCE
from tlt.textutil import dominant_script
from tlt.stages.s04_generate import current_generations
from tlt.stages.s05_mask_and_judge import targets_from_generations
from tlt.stages.s07_translated_judging import back_translate, eligible
from tlt.translate import make_translator

run = load_run("full")
gens = current_generations(run)
prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
pool = [t for t in targets_from_generations(gens, prompts) if eligible(t)]
share = float(run.design["translated_judging"]["share"])
strata = defaultdict(list)
for t in pool:
    strata[(t["block"], t["condition"], t["model"])].append(t)
rng = random.Random(run.seed + 7)
sample = []
for k in sorted(strata):
    grp = sorted(strata[k], key=lambda t: t["target_id"])
    sample.extend(rng.sample(grp, max(1, round(share * len(grp)))))
print("sample", len(sample), flush=True)
llm, ulog = make_llm(run), AppendLog(run.path("runs", "utility_calls.jsonl"))
# Code-mixed replies: the utility model (no GPU), the same calls stage 7 makes, run in parallel
# so their cached results are ready; started first so they overlap the GPU work below.
cm = [t for t in sample if t["condition"] in CM_SOURCE and dominant_script(mask_names(t["response_text"])[0]) == "latin"]
pool = ThreadPoolExecutor(12)
futs = [pool.submit(protected_utility, llm, ulog, f"tj_backtr|{t['target_id']}", "utility_translate_to_en", "text",
                    mask_names(t["response_text"])[0],
                    {"role": "translated_judging_backtr", "target_id": t["target_id"]},
                    source_description=CM_SOURCE[t["condition"]]) for t in cm]
print("code-mixed submitted", len(cm), flush=True)
# Indic replies: IndicTrans2 on the GPU, via stage 7's own function (code-mixed ones excluded).
rest = [t for t in sample if t not in cm]
back_translate(run, make_translator(run), llm, ulog, rest)
print("GPU PART DONE: indic translated", sum(bool((t.get("translated_text") or "").strip()) for t in rest),
      "of", len(rest), flush=True)
ok = sum(1 for f in futs if (f.result()[0] or "").strip())
pool.shutdown()
print("code-mixed done", ok, "of", len(cm), flush=True)

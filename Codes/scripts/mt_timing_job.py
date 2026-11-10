#!/usr/bin/env python3
"""GPU timing of IndicTrans2 for single messages (E12; runs on the rented Vast box).

    HF_HUB_OFFLINE=1 GAPTRACE_ATTN=flash_attention_2 python mt_timing_job.py jobs.jsonl out.jsonl

Each job is the word segments of ONE message (as in an interactive call), translated in one
batch with the pipeline's decoding (fp16, beam 5, max length 256). Five warm-up jobs per model,
then every job is timed three times with CUDA synchronisation; the median is reported.
"""
import json
import os
import statistics
import sys
import time

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ATTN = os.environ.get("GAPTRACE_ATTN", "sdpa")
DEV = "cuda:0"
jobs = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
out = open(sys.argv[2], "w", encoding="utf-8")
name = torch.cuda.get_device_name(0)


def run(tok, model, ip, job):
    if not job["segments"]:
        return [], 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = ip.preprocess_batch(job["segments"], src_lang=job["src"], tgt_lang=job["tgt"])
    enc = tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEV)
    with torch.inference_mode():
        gen = model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5,
                             num_return_sequences=1, do_sample=False)
    dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    res = ip.postprocess_batch(dec, lang=job["tgt"])
    torch.cuda.synchronize()
    return res, time.perf_counter() - t0


for model_id in sorted({j["model"] for j in jobs}):
    mine = [j for j in jobs if j["model"] == model_id]
    rev = mine[0]["revision"]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=rev)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True, revision=rev,
                                                  torch_dtype=torch.float16, attn_implementation=ATTN).to(DEV).eval()
    ip = IndicProcessor(inference=True)
    for j in mine[:5]:
        run(tok, model, ip, j)
    for j in mine:
        times, res = [], None
        for _ in range(3):
            r, s = run(tok, model, ip, j)
            res = res if res is not None else r
            times.append(s)
        out.write(json.dumps({"k": j["k"], "median_s": statistics.median(times), "times": times,
                              "segments": len(j["segments"]), "chars": j["chars"],
                              "device": f"{name} float16 {ATTN}", "sample_output": (res or [""])[0][:200]},
                             ensure_ascii=False) + "\n")
        out.flush()
    print(model_id, "done", len(mine), flush=True)
    del model
    torch.cuda.empty_cache()
print("ALL_DONE", flush=True)

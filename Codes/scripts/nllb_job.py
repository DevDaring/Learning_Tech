"""Remote batch job for E8: translate English segments to Hindi/Bengali with NLLB-200 on a GPU.

    python nllb_job.py segments.jsonl out.jsonl
Input lines: {"k": key, "lang": "HI"|"BN", "text": segment}. Output lines: {"k": key, "v": text}.
Deterministic beam search (5 beams), the same settings as the IndicTrans2 path.
"""
import json
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-distilled-1.3B"
TAGS = {"HI": "hin_Deva", "BN": "ben_Beng"}
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in open(src, encoding="utf-8")]
try:  # resume: keep what an earlier (interrupted) run already wrote
    done = {json.loads(x)["k"] for x in open(dst, encoding="utf-8") if x.strip()}
except (FileNotFoundError, json.JSONDecodeError):
    done = set()
rows = [r for r in rows if r["k"] not in done]
tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
out = open(dst, "a", encoding="utf-8")
for lang, tag in TAGS.items():
    sub = sorted([r for r in rows if r["lang"] == lang], key=lambda r: len(r["text"]))
    for i in range(0, len(sub), 16):
        b = sub[i:i + 16]
        enc = tok([r["text"] for r in b], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tag), num_beams=5,
                                 max_length=256)
        for r, t in zip(b, tok.batch_decode(gen, skip_special_tokens=True)):
            out.write(json.dumps({"k": r["k"], "v": t}, ensure_ascii=False) + "\n")
        out.flush()
        print(lang, min(i + 16, len(sub)), "/", len(sub), flush=True)
out.close()
print("DONE", flush=True)

#!/usr/bin/env python3
"""IndicTrans2 translation service for a rented GPU (run there, reached via an SSH tunnel).

    TLT_TOKEN=<random> python indictrans_server.py --port 8000

POST /translate  {"model": <hf id>, "revision": <commit>, "src": "eng_Latn", "tgt": "hin_Deva",
                  "segments": [...], "num_beams": 5, "max_length": 256}
  -> {"translations": [...], "device": "cuda:0 NVIDIA ...", "dtype": "float32"}
GET  /health     -> {"ok": true, "models": [...]}

Same pre/post-processing and decoding as the local CPU path in src/tlt/translate.py
(IndicProcessor, beam search, no sampling). On the GPU it runs float16 with FlashAttention-2,
as the IndicTrans2 model card recommends; the device, dtype and attention backend are
returned with every response and cached with every translation. Listens on 127.0.0.1 only; every request must carry the X-Token header.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TOKEN = os.environ["TLT_TOKEN"]
DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[os.environ.get("TLT_DTYPE", "float16")]
ATTN = os.environ.get("TLT_ATTN", "flash_attention_2")
MODELS: dict[str, tuple] = {}
LOCK = threading.Lock()
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(model_id: str, revision: str | None):
    key = f"{model_id}@{revision}"
    if key not in MODELS:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True, revision=revision,
                                                      torch_dtype=DTYPE, attn_implementation=ATTN).to(DEV).eval()
        MODELS[key] = (tok, model, IndicProcessor(inference=True))
    return MODELS[key]


def translate(req: dict) -> list[str]:
    tok, model, ip = load(req["model"], req.get("revision"))
    segs = req["segments"]
    order = sorted(range(len(segs)), key=lambda i: len(segs[i]))  # length-sorted batches
    out = [""] * len(segs)
    bs = int(req.get("batch_size", 32))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        batch = ip.preprocess_batch([segs[i] for i in idx], src_lang=req["src"], tgt_lang=req["tgt"])
        enc = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                  return_attention_mask=True).to(DEV)
        with torch.inference_mode():
            gen = model.generate(**enc, use_cache=True, min_length=0, max_length=int(req.get("max_length", 256)),
                                 num_beams=int(req.get("num_beams", 5)), num_return_sequences=1, do_sample=False)
        dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for i, o in zip(idx, ip.postprocess_batch(dec, lang=req["tgt"])):
            out[i] = o
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, {"error": "forbidden"})
        self._send(200, {"ok": True, "device": f"{DEV} {str(DTYPE).split('.')[-1]} {ATTN}", "models": list(MODELS)})

    def do_POST(self):
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, {"error": "forbidden"})
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:  # one batch at a time on the GPU
                res = translate(req)
            name = torch.cuda.get_device_name(0) if DEV.startswith("cuda") else "cpu"
            self._send(200, {"translations": res, "device": f"{DEV} {name} {str(DTYPE).split('.')[-1]} {ATTN}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", nargs="*", default=[], help="model@revision to load at start")
    a = ap.parse_args()
    for spec in a.preload:
        m, _, r = spec.partition("@")
        load(m, r or None)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

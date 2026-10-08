"""IndicTrans2 wrapper (deterministic beam search) with maths protection and a disk cache.

Long texts are split into lines and sentences (IndicTrans2 is sentence-level, max 256
tokens), translated, and re-joined with the original line structure. LaTeX spans are
replaced by <IDn> placeholders before translation and restored afterwards; a span that
does not survive is reported so the caller can flag the item.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time

from . import secrets
from .config import CODES_DIR, Run, stable_hash
from .textutil import has_words, protect_math, restore_math

log = logging.getLogger("tlt.translate")

run_token_name = "HUGGINGFACE_TOKEN"
_SENT_SPLIT = re.compile(r"(?<=[.?!।॥])\s+")


def split_segments(text: str, max_chars: int = 400) -> list[tuple[int, str]]:
    """(line_no, segment) pairs; empty lines are kept as empty segments."""
    out = []
    for ln, line in enumerate((text or "").split("\n")):
        if not line.strip():
            out.append((ln, ""))
            continue
        for s in _SENT_SPLIT.split(line.strip()):
            while len(s) > max_chars:  # no sentence boundary: break at a comma or space
                cut = max(s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
                cut = cut if cut > 50 else max_chars
                out.append((ln, s[:cut + 1].strip()))
                s = s[cut + 1:]
            if s.strip():
                out.append((ln, s.strip()))
    return out


def join_segments(pairs: list[tuple[int, str]]) -> str:
    lines: dict[int, list[str]] = {}
    for ln, s in pairs:
        lines.setdefault(ln, [])
        if s:
            lines[ln].append(s)
    return "\n".join(" ".join(lines[k]) for k in sorted(lines))


class IndicTrans2:
    def __init__(self, run: Run):
        cfg = run.models["translation"]
        self.cfg = cfg
        self.tags = cfg["lang_tags"]
        self.model_id_en_indic = cfg["en_indic"]
        self.model_id_indic_en = cfg["indic_en"]
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._tlock = threading.Lock()  # one translation at a time: shared model + cache file
        self.cache_path = run.path("data", "cache", "indictrans2.jsonl")
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["v"]
                except (json.JSONDecodeError, KeyError):
                    pass

    def _load(self, model_id: str):
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            import gc
            import torch
            # keep one 1B model resident (about 4.5 GB in fp32); stages batch by direction
            for other in list(self._models):
                del self._models[other]
            gc.collect()
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            token = secrets.get(run_token_name)
            dev = self.cfg.get("device", "auto")
            dev = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
            rev = (self.cfg.get("revisions") or {}).get(model_id)
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token, revision=rev)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True, token=token, revision=rev,
                torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            self._models[model_id] = (tok, model, IndicProcessor(inference=True), dev)
            log.info("loaded %s on %s", model_id, dev)
            return self._models[model_id]

    def _remote(self) -> tuple[str, str] | None:
        """(url, token) of a GPU translation service (scripts/vast_translator.py), if active."""
        import os
        url = os.environ.get("TLT_TRANSLATE_URL")
        if not url:
            return None
        tok = os.environ.get("TLT_TRANSLATE_TOKEN")
        if not tok:
            st = CODES_DIR / "runs" / "vast_translator.json"
            tok = json.loads(st.read_text()).get("token") if st.exists() else None
        return url.rstrip("/"), tok

    def _translate_remote(self, segs: list[str], src: str, tgt: str, model_id: str) -> tuple[list[str], str]:
        import requests
        url, tok = self._remote()
        body = {"model": model_id, "revision": (self.cfg.get("revisions") or {}).get(model_id), "src": src,
                "tgt": tgt, "segments": segs, "num_beams": int(self.cfg["num_beams"]),
                "max_length": int(self.cfg["max_length"]), "batch_size": 32}
        for attempt in range(5):
            try:
                r = requests.post(f"{url}/translate", json=body, headers={"X-Token": tok or ""}, timeout=1800)
                if r.ok:
                    d = r.json()
                    return d["translations"], d.get("device", "remote")
                err = r.text[:200]
            except requests.RequestException as e:
                err = type(e).__name__
            log.warning("remote translation failed (%s); retry %d", err, attempt + 1)
            time.sleep(10 * (attempt + 1))
        raise RuntimeError("remote translation service unavailable: " + err)

    def _translate_segments(self, segs: list[str], src: str, tgt: str, model_id: str) -> list[str]:
        keys = [stable_hash(model_id, src, tgt, s, n=24) for s in segs]
        todo = [i for i, k in enumerate(keys) if k not in self.cache and segs[i]]
        if todo and self._remote():
            with open(self.cache_path, "a", encoding="utf-8") as cache_f:
                for b in range(0, len(todo), 512):
                    idx = todo[b:b + 512]
                    outs, device = self._translate_remote([segs[i] for i in idx], src, tgt, model_id)
                    for i, o in zip(idx, outs):
                        self.cache[keys[i]] = o
                        cache_f.write(json.dumps({"k": keys[i], "v": o, "device": device}, ensure_ascii=False) + "\n")
                    log.info("remote-translated %d/%d segments (%s)", min(b + 512, len(todo)), len(todo), device)
            todo = []
        if todo:
            import torch
            tok, model, ip, dev = self._load(model_id)
            bs = int(self.cfg["batch_size"])
            with open(self.cache_path, "a", encoding="utf-8") as cache_f:
                for b in range(0, len(todo), bs):
                    idx = todo[b:b + bs]
                    batch = ip.preprocess_batch([segs[i] for i in idx], src_lang=src, tgt_lang=tgt)
                    enc = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                              return_attention_mask=True).to(dev)
                    with torch.inference_mode():
                        gen = model.generate(**enc, use_cache=True, min_length=0,
                                             max_length=int(self.cfg["max_length"]),
                                             num_beams=int(self.cfg["num_beams"]), num_return_sequences=1,
                                             do_sample=False)
                    dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    outs = ip.postprocess_batch(dec, lang=tgt)
                    for i, o in zip(idx, outs):
                        self.cache[keys[i]] = o
                        cache_f.write(json.dumps({"k": keys[i], "v": o}, ensure_ascii=False) + "\n")
        return [self.cache.get(k, "") if s else "" for k, s in zip(keys, segs)]

    def _translate(self, texts: list[str], src: str, tgt: str, model_id: str) -> list[tuple[str, list[str]]]:
        """Returns (translation, missing_math_spans) per text."""
        with self._tlock:
            return self._translate_locked(texts, src, tgt, model_id)

    def _translate_locked(self, texts, src, tgt, model_id):
        plans, flat = [], []
        for t in texts:
            protected, mapping = protect_math(t)
            if not has_words(protected):  # pure maths: nothing to translate, kept verbatim
                plans.append(None)
                continue
            pairs = split_segments(protected)
            # segments without words (a maths block, a table row) are never sent to the model
            plans.append((pairs, mapping, len(flat)))
            flat.extend(s if has_words(s) else "" for _, s in pairs)
        outs = self._translate_segments(flat, src, tgt, model_id)
        results = []
        for t, plan in zip(texts, plans):
            if plan is None:
                results.append((t, []))
                continue
            pairs, mapping, start = plan
            tr = [(ln, outs[start + j] if has_words(s) else s) for j, (ln, s) in enumerate(pairs)]
            joined, missing = restore_math(join_segments(tr), mapping)
            results.append((joined, missing))
        return results

    def en_to_indic(self, texts: list[str], lang: str) -> list[str]:
        return [t for t, _ in self.en_to_indic_checked(texts, lang)]

    def en_to_indic_checked(self, texts: list[str], lang: str) -> list[tuple[str, list[str]]]:
        return self._translate(texts, self.tags["EN"], self.tags[lang], self.model_id_en_indic)

    def indic_to_en(self, texts: list[str], lang: str) -> list[str]:
        return [t for t, _ in self.indic_to_en_checked(texts, lang)]

    def indic_to_en_checked(self, texts: list[str], lang: str) -> list[tuple[str, list[str]]]:
        return self._translate(texts, self.tags[lang], self.tags["EN"], self.model_id_indic_en)


def make_translator(run: Run):
    if run.dry:
        from .mock import MockTranslator

        class _Mock(MockTranslator):
            def en_to_indic_checked(self, texts, lang):
                return [(t, []) for t in self.en_to_indic(texts, lang)]

            def indic_to_en_checked(self, texts, lang):
                return [(t, []) for t in self.indic_to_en(texts, lang)]
        return _Mock()
    return IndicTrans2(run)


class Embedder:
    def __init__(self, run: Run):
        self.model_id = run.models["embedding"]["model"]
        self.revision = run.models["embedding"].get("revision")
        self.bs = int(run.models["embedding"]["batch_size"])
        self._m = None

    def encode(self, texts: list[str]):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_id, token=secrets.get("HUGGINGFACE_TOKEN"),
                                          revision=self.revision)
        return self._m.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 200, convert_to_numpy=True)


def make_embedder(run: Run):
    if run.dry:
        from .mock import MockEmbedder
        return MockEmbedder()
    return Embedder(run)


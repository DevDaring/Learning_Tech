"""Command line: ``python -m tlt <stage> [--pilot | --dry-run]`` (run from Codes/ with PYTHONPATH=src)."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time

from .config import load_run

STAGES = {
    "verify": "s00_verify",
    "build_items": "s01_build_items",
    "translate": "s02_translate",
    "build_prompts": "s03_build_prompts",
    "generate": "s04_generate",
    "mask_and_judge": "s05_mask_and_judge",
    "judge_validation": "s06_judge_validation",
    "translated_judging": "s07_translated_judging",
    "auto_metrics": "s08_auto_metrics",
    "analysis": "s09_analysis",
    "baselines": "s10_baselines",
    "repair": "s11_repair",
    "export_human": "s12_export_human",
    "checks": "checks",
    "review_pack": "review_pack",
    "import_review": "import_review",
    "estimate_cost": "estimate_cost",
    "prepilot": "prepilot",
}

# Dependency order for `all`: repair precedes auto_metrics and baselines because B5 and the
# repair table need the repaired replies; analysis comes after every result exists.
ORDER = ["build_items", "translate", "build_prompts", "generate", "mask_and_judge", "judge_validation",
         "translated_judging", "repair", "auto_metrics", "baselines", "export_human", "analysis", "checks"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tlt", description=__doc__)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true", help="30 items, 1 sample, 2 producers, 3 judges (real calls)")
    g.add_argument("--dry-run", action="store_true", help="pilot counts with mock models (no network)")
    g.add_argument("--smoke", action="store_true", help="2 items per block, all models, every stage (real calls)")
    ap.add_argument("--only", nargs="*", help="baselines: subset of B1 B2 B3 B5; verify: model keys")
    ap.add_argument("--files", nargs="*", help="import_review: returned review workbooks (.xlsx)")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="full run despite outstanding human gates; outputs are marked PRELIMINARY")
    args = ap.parse_args(argv)
    if args.allow_unreviewed:
        import os
        os.environ["TLT_ALLOW_UNREVIEWED"] = "1"
    mode = "dryrun" if args.dry_run else "smoke" if args.smoke else "pilot" if args.pilot else "full"
    run = load_run(mode)
    log_path = run.path("runs", "pipeline.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, encoding="utf-8")])
    for noisy in ("urllib3", "botocore", "boto3", "httpx", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    stages = ([s for s in ORDER] if args.stage == "all" else [args.stage])
    if args.stage == "all" and mode != "dryrun":
        stages = ["verify"] + stages
    summary = {}
    for s in stages:
        t0 = time.time()
        mod = importlib.import_module(f".stages.{STAGES[s]}", __package__)
        kwargs = {}
        if args.only and s == "baselines":
            kwargs["only"] = args.only
        if args.only and s == "verify":
            kwargs["models"] = args.only
        if s == "prepilot" and args.only:
            kwargs["phase"] = args.only[0]
        if s == "import_review":
            kwargs["files"] = args.files
        res = mod.run_stage(run, **kwargs)
        summary[s] = res
        logging.getLogger("tlt").info("stage %s done in %.1fs", s, time.time() - t0)
    out = run.path("runs", f"last_summary_{args.stage}.json")
    out.write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    if args.stage == "checks" and not summary["checks"].get("all_pass"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

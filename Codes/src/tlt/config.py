"""Configuration, run modes and output paths.

Three run modes share one code path and differ only in their output root and counts:
  full     -> Codes/{data,runs,results,review}
  pilot    -> Codes/pilot/{data,runs,results,review}  (30 items, 1 sample, 2 producers)
  dry-run  -> Codes/dryrun/...  (pilot counts, mock LLM and mock translation; no network)
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CODES_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = CODES_DIR / "config"
PROMPTS_DIR = CODES_DIR / "prompts"
DOCS_DIR = CODES_DIR / "docs"


def _deep_update(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Run:
    """Everything a stage needs: design, model config, mode and paths."""
    design: dict
    models: dict
    mode: str  # "full" | "pilot" | "dryrun"

    @property
    def pilot(self) -> bool:
        return self.mode in ("pilot", "dryrun", "smoke")

    @property
    def dry(self) -> bool:
        return self.mode == "dryrun"

    @property
    def root(self) -> Path:
        return {"full": CODES_DIR, "pilot": CODES_DIR / "pilot", "dryrun": CODES_DIR / "dryrun",
                "prepilot": CODES_DIR / "prepilot", "smoke": CODES_DIR / "smoke"}[self.mode]

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def source(self, rel: str) -> Path:
        """Input datasets always live under Codes/, whatever the mode."""
        return CODES_DIR / rel

    @property
    def seed(self) -> int:
        return int(self.design["seed"])

    @property
    def producers(self) -> list[str]:
        return list(self.design["producers"])

    @property
    def judges(self) -> list[str]:
        return list(self.design["judges"])

    @property
    def provisional(self) -> bool:
        """Main results stay provisional until the human-anchor file has come back."""
        return not any(self.path("review", "human_returned", "x").parent.glob("*.csv")) or self.pilot


def load_run(mode: str = "full") -> Run:
    design = load_yaml(CONFIG_DIR / "design.yaml")
    models = load_yaml(CONFIG_DIR / "models.yaml")
    if mode in ("pilot", "dryrun"):
        design = _deep_update(copy.deepcopy(design), design.get("pilot", {}))
    elif mode == "smoke":
        design = _deep_update(copy.deepcopy(design), design.get("smoke", {}))
    return Run(design=design, models=models, mode=mode)


def stable_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:n]


def stable_int(*parts: Any) -> int:
    return int(stable_hash(*parts, n=8), 16) & 0x7FFFFFFF

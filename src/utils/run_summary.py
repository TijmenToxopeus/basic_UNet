# src/utils/run_summary.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, obj: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)
    return path


def base_run_info(cfg: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
    """
    Shared metadata across train/eval/prune. Keep this stable.
    """
    exp = cfg.get("experiment", {})
    return {
        "schema_version": 1,
        "stage": stage,  # "train" | "eval" | "prune"
        "created_utc": now_utc_iso(),
        "experiment": {
            "experiment_name": exp.get("experiment_name"),
            "model_name": exp.get("model_name"),
            "seed": exp.get("seed"),
            "deterministic": exp.get("deterministic"),
            "device": exp.get("device"),
        },
    }


def attach_profile(summary: Dict[str, Any], prof: Dict[str, float]) -> None:
    """
    prof expected from profile_model().
    """
    summary["profile"] = {
        "params_m": float(prof.get("params_m", float("nan"))),
        "flops_g": float(prof.get("flops_g", float("nan"))),
        "inference_ms": float(prof.get("inference_ms", float("nan"))),
    }

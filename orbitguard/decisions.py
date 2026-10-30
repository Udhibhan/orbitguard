"""Append-only analyst decision log (data/decisions.jsonl). The human decides; we record."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import data

LOG = data.ROOT / "data" / "decisions.jsonl"


def log_decision(record: dict, path=LOG) -> dict:
    rec = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def load_decisions(path=LOG) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "dataset", "event_id", "decision"])
    return pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])

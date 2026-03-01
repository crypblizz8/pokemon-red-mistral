from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_payload() -> Dict[str, object]:
    return {
        "version": 1,
        "updated_at": "",
        "totals": {
            "real_battles": 0,
            "simulations": 0,
            "movement_steps": 0,
            "phase3_steps": 0,
            "phase4_steps": 0,
        },
        "entries": [],
    }


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _load_payload(path: Path) -> Dict[str, object]:
    if not path.exists():
        return _new_payload()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _new_payload()

    if not isinstance(data, dict):
        return _new_payload()

    totals = data.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []

    return {
        "version": _safe_int(data.get("version")) or 1,
        "updated_at": str(data.get("updated_at") or ""),
        "totals": {
            "real_battles": _safe_int(totals.get("real_battles")),
            "simulations": _safe_int(totals.get("simulations")),
            "movement_steps": _safe_int(totals.get("movement_steps")),
            "phase3_steps": _safe_int(totals.get("phase3_steps")),
            "phase4_steps": _safe_int(totals.get("phase4_steps")),
        },
        "entries": entries,
    }


def append_campaign_log_entry(
    path: Path,
    *,
    kind: str,
    count: int,
    source: str,
    movement_steps: int = 0,
    phase: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if kind not in {"real_battle", "simulation"}:
        raise ValueError(f"Unsupported log kind: {kind}")
    if count <= 0:
        return _load_payload(path)

    payload = _load_payload(path)
    totals = payload["totals"]
    if not isinstance(totals, MutableMapping):
        totals = {
            "real_battles": 0,
            "simulations": 0,
            "movement_steps": 0,
            "phase3_steps": 0,
            "phase4_steps": 0,
        }

    counter_key = "real_battles" if kind == "real_battle" else "simulations"
    totals[counter_key] = _safe_int(totals.get(counter_key)) + int(count)
    normalized_steps = max(0, _safe_int(movement_steps))
    totals["movement_steps"] = _safe_int(totals.get("movement_steps")) + normalized_steps
    phase_name = str(phase or "").strip().lower()
    if not phase_name and isinstance(metadata, Mapping):
        phase_name = str(metadata.get("phase") or "").strip().lower()
    if phase_name == "phase3":
        totals["phase3_steps"] = _safe_int(totals.get("phase3_steps")) + normalized_steps
    elif phase_name == "phase4":
        totals["phase4_steps"] = _safe_int(totals.get("phase4_steps")) + normalized_steps
    payload["totals"] = totals

    entry: Dict[str, object] = {
        "timestamp": utc_now_iso(),
        "kind": kind,
        "count": int(count),
        "source": source,
    }
    if normalized_steps > 0:
        entry["movement_steps"] = normalized_steps
    if phase_name:
        entry["phase"] = phase_name
    if metadata:
        entry["metadata"] = dict(metadata)

    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    entries.append(entry)
    payload["entries"] = entries
    payload["updated_at"] = utc_now_iso()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def campaign_log_report(path: Path) -> Dict[str, object]:
    payload = _load_payload(path)
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    real_battles = _safe_int(totals.get("real_battles"))
    simulations = _safe_int(totals.get("simulations"))
    movement_steps = _safe_int(totals.get("movement_steps"))
    phase3_steps = _safe_int(totals.get("phase3_steps"))
    phase4_steps = _safe_int(totals.get("phase4_steps"))
    entries = payload.get("entries")
    return {
        "path": str(path.resolve()),
        "real_battles": real_battles,
        "simulations": simulations,
        "combined": real_battles + simulations,
        "movement_steps": movement_steps,
        "phase3_steps": phase3_steps,
        "phase4_steps": phase4_steps,
        "updated_at": str(payload.get("updated_at") or ""),
        "entries": len(entries) if isinstance(entries, list) else 0,
    }

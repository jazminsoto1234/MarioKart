"""Etapa 0b: inventario de episodios (p. ej. del Drive), auditoría y split por grupo.

La descarga se hace fuera (Google Drive para escritorio o rclone), p. ej.:
    rclone copy drive:MarioKart data/drive --include "*/meta.json" --include "*/READY.json"
    rclone copy drive:MarioKart data/drive --include "*/data.h5" --include "*/track.npz"
"""
from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from stk_agent.audit import audit_path
from stk_agent.util import atomic_json

COLUMNS = ("path", "group", "episode_id", "driver_id", "track", "reverse", "difficulty", "num_karts",
           "laps", "input_device", "seconds", "width", "height", "tick_hz", "rgb_hz", "schema_version",
           "finished", "has_ready", "has_data", "has_video", "has_track")


def episode_row(meta_path: Path, root: Path) -> dict:
    ep = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    s = meta.get("settings", {})
    rel = ep.relative_to(root)
    return {"path": str(ep), "group": rel.parts[0] if len(rel.parts) > 1 else meta.get("group_id", ""),
            "episode_id": meta.get("episode_id", ep.name), "driver_id": meta.get("driver_id"),
            "track": s.get("track"), "reverse": s.get("reverse"), "difficulty": s.get("difficulty"),
            "num_karts": s.get("num_karts"), "laps": s.get("laps"), "input_device": meta.get("input_device"),
            "seconds": meta.get("simulated_seconds"), "width": s.get("width"), "height": s.get("height"),
            "tick_hz": s.get("tick_hz"), "rgb_hz": s.get("rgb_hz"), "schema_version": meta.get("schema_version"),
            "finished": (meta.get("result") or {}).get("finished"),
            "has_ready": (ep / "READY.json").exists(), "has_data": (ep / "data.h5").exists(),
            "has_video": (ep / "rgb.mp4").exists(), "has_track": (ep / "track.npz").exists()}


def inventory(root: str, out_csv: str) -> list[dict]:
    """Una fila por episodio (meta.json), deduplicado por episode_id."""
    root_p = Path(root)
    rows, seen = [], set()
    for meta_path in sorted(root_p.rglob("meta.json")):
        row = episode_row(meta_path, root_p)
        if row["episode_id"] in seen:
            continue
        seen.add(row["episode_id"])
        rows.append(row)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    usable = [r for r in rows if is_usable(r)]
    hours = sum(float(r["seconds"] or 0) for r in usable) / 3600
    print(f"[ingest] {len(rows)} episodios, {len(usable)} utilizables ({hours:.1f} h), "
          f"{len({r['group'] for r in rows})} grupos → {out_csv}")
    return rows


def is_usable(r: dict) -> bool:
    """Completo y con el contrato esperado (640×360, 60/30 Hz)."""
    def truthy(v):
        return v is True or str(v).lower() == "true"
    return (all(truthy(r[k]) for k in ("has_ready", "has_data", "has_video", "has_track"))
            and str(r["width"]) == "640" and str(r["height"]) == "360"
            and str(r["tick_hz"]) == "60" and str(r["rgb_hz"]) == "30")


def audit(root: str, out_json: str):
    report = audit_path(root)
    atomic_json(out_json, report)
    print(f"[ingest] audit: válidos {report.get('episodes_valid', report.get('valid'))} → {out_json}")
    return report


def make_split(inventory_csv: str, out_json: str, heldout_tracks: list[str],
               val_groups: float = 0.15, seed: int = 0, audit_json: str = "") -> dict:
    """Split por GRUPO: grupos completos a validación; pistas reservadas fuera de train/val."""
    with open(inventory_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if is_usable(r)]
    if audit_json:
        report = json.loads(Path(audit_json).read_text(encoding="utf-8"))
        bad = {e["path"] for e in report.get("episodes", []) if not e.get("valid")}
        rows = [r for r in rows if r["path"] not in bad]
    groups = sorted({r["group"] for r in rows})
    n_val = max(1, int(round(len(groups) * val_groups))) if len(groups) > 1 else 0
    val = set(np.random.default_rng(seed).permutation(groups)[:n_val])
    held = set(heldout_tracks)
    split = {"train": [r["path"] for r in rows if r["group"] not in val and r["track"] not in held],
             "val": [r["path"] for r in rows if r["group"] in val and r["track"] not in held],
             "heldout_track": [r["path"] for r in rows if r["track"] in held],
             "val_groups": sorted(val)}
    atomic_json(out_json, split)
    print(f"[ingest] split: train {len(split['train'])}, val {len(split['val'])} "
          f"({len(val)} grupos), excluidos por pista {len(split['heldout_track'])} → {out_json}")
    return split

from pathlib import Path
import json
import h5py
import numpy as np
from .util import sha256, safe_relative
from .video import iter_video
SCHEMA_VERSION = "1.0.0"

def validate_episode(path, verify_hashes=True, decode_video=False, allow_partial=False):
    path = Path(path)
    errors = []
    report = {"path": str(path), "valid": False, "errors": errors}
    try:
        if not allow_partial and (path.name.endswith(".partial") or not (path/"READY.json").is_file()):
            raise ValueError("Falta READY.json o el episodio sigue siendo .partial")
        meta = json.loads((path/"meta.json").read_text(encoding="utf-8"))
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Versión de esquema no soportada")
        n, t = meta["num_frames"], meta["num_transitions"]
        if n != t+1 or t < 1:
            raise ValueError("Se requieren T>=1 acciones y T+1 observaciones")
        if verify_hashes:
            hashes = json.loads((path/"checksums.json").read_text(encoding="utf-8"))
            required = {"meta.json", "data.h5", "track.npz", meta["rgb_file"]}
            if not required.issubset(hashes):
                raise ValueError("Faltan archivos en checksums.json")
            for name, expected in hashes.items():
                if sha256(safe_relative(path, name)) != expected:
                    raise ValueError(f"SHA256 incorrecto: {name}")
            if (path/"READY.json").exists():
                ready = json.loads((path/"READY.json").read_text())
                if ready["checksums_sha256"] != sha256(path/"checksums.json"):
                    raise ValueError("READY/checksums no coinciden")
        with h5py.File(path/"data.h5", "r") as f:
            if not np.array_equal(f["frames/id"][:], np.arange(n)):
                raise ValueError("IDs de frames no contiguos")
            if not np.array_equal(f["actions/frame_id"][:], np.arange(t)):
                raise ValueError("IDs de acciones no contiguos")
            for k in ("value", "terminated", "truncated", "input_time_ns", "applied_time_ns"):
                if len(f["actions/"+k]) != t:
                    raise ValueError(f"Cantidad inconsistente de acciones/{k}")
            for k in ("sim_time", "display_time_ns"):
                if len(f["frames/"+k]) != n:
                    raise ValueError(f"Cantidad inconsistente de frames/{k}")
            times = f["frames/sim_time"][:]
            if not np.allclose(times, np.arange(n)/meta["settings"]["tick_hz"], rtol=0, atol=1e-8):
                raise ValueError("Timestamps simulados incompatibles")
            a = f["actions/value"][:]
            if a.shape != (t, 7) or not np.isfinite(a).all():
                raise ValueError("Acciones con shape incorrecto o NaN/Inf")
            if np.any(np.abs(a[:, 0]) > 1) or np.any((a[:, 1] < 0) | (a[:, 1] > 1)):
                raise ValueError("Acciones continuas fuera de rango")
            if not np.isin(a[:, 2:], [0, 1]).all():
                raise ValueError("Acciones binarias inválidas")
            if f["actions/terminated"][:-1].any() or f["actions/truncated"][:-1].any():
                raise ValueError("Hay una terminación antes del último paso")
            offsets = f["privileged/state_offsets"][:]
            if len(offsets) != n+1 or offsets[0] != 0 or (np.diff(offsets) <= 0).any():
                raise ValueError("Offsets del estado inválidos")
            if offsets[-1] != len(f["privileged/state_bytes"]):
                raise ValueError("Estado truncado")
            ids = f["aux/frame_id"][:]
            if len(ids) != meta["num_aux_frames"] or (np.diff(ids) <= 0).any():
                raise ValueError("Índices auxiliares inconsistentes")
            if len(ids) and (ids.min() < 0 or ids.max() >= n):
                raise ValueError("Frame auxiliar inexistente")
            for name in ("depth", "instance"):
                if f["aux/"+name].shape != (len(ids), meta["settings"]["height"], meta["settings"]["width"]):
                    raise ValueError("Dimensiones auxiliares inválidas")
        if decode_video:
            settings = meta["settings"]
            decoded = sum(1 for _ in iter_video(safe_relative(path, meta["rgb_file"]), settings["width"], settings["height"]))
            if decoded != n:
                raise ValueError(f"Video: {decoded} frames, se esperaban {n}")
        report.update(valid=True, frames=n, transitions=t, source=meta.get("source"),
                      episode_id=meta["episode_id"], simulated_seconds=meta["simulated_seconds"])
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
    return report

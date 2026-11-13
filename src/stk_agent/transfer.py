from __future__ import annotations
from pathlib import Path
import json
import os
import shutil
import zipfile
from .dataset import discover_episodes
from .recording import validate_episode
from .util import atomic_json, sha256, safe_relative


def publish(episode, drive_dir):
    source = Path(episode).resolve()
    report = validate_episode(source, verify_hashes=True)
    if not report["valid"]:
        raise ValueError(report["errors"])
    destination_root = Path(drive_dir).resolve()
    # Copiar en sí mismo o en sus hijos sería incorrecto.
    if destination_root == source or source in destination_root.parents:
        raise ValueError("El destino no puede estar dentro del episodio origen")
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root/report["episode_id"]
    if target.exists():
        existing = validate_episode(target, verify_hashes=True)
        if existing["valid"] and sha256(target/"checksums.json") == sha256(source/"checksums.json"):
            return target
        raise FileExistsError(f"Conflicto: {target}; no se sobreescribe")
    staging = target.with_name(target.name+".uploading")
    staging.mkdir(exist_ok=False)
    hashes = json.loads((source/"checksums.json").read_text(encoding="utf-8"))
    for name in [*hashes, "checksums.json"]:
        src = safe_relative(source, name)
        dst = safe_relative(staging, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for name, expected in hashes.items():
        if sha256(safe_relative(staging, name)) != expected:
            raise IOError(f"Error de copia: {name}; conservado {staging}")
    os.replace(staging, target)
    # READY al final. Drive podría sincronizar fuera de orden; el lector valida hashes.
    shutil.copy2(source/"READY.json", target/"READY.json")
    return target


def pack_episode(episode, output):
    episode, output = Path(episode), Path(output)
    report = validate_episode(episode, verify_hashes=True)
    if not report["valid"]:
        raise ValueError(report["errors"])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((episode/"checksums.json").read_text(encoding="utf-8"))
    # Ya comprimidos; ZIP_STORED evita comprimir otra vez GB de videos y HDF5.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name in [*hashes, "checksums.json", "READY.json"]:
            z.write(safe_relative(episode, name), f"{report['episode_id']}/{name}")
    return output


def build_index(root, output):
    rows, rejected = [], []
    for path in discover_episodes(root):
        result = validate_episode(path, verify_hashes=True)
        if not result["valid"]:
            rejected.append(result)
            continue
        meta = json.loads((path/"meta.json").read_text(encoding="utf-8"))
        rows.append({"path": str(path.resolve()), "episode_id": meta["episode_id"],
                     "group_id": meta.get("group_id"), "driver_id": meta.get("driver_id"),
                     "source": meta.get("source"), "track": meta["settings"]["track"],
                     "reverse": meta["settings"]["reverse"], "seed": meta["settings"]["seed"],
                     "transitions": meta["num_transitions"], "seconds": meta["simulated_seconds"],
                     "result": meta["result"], "quality": meta.get("quality", {})})
    atomic_json(output, {"episodes": rows, "rejected": rejected,
                         "total_seconds": sum(x["seconds"] for x in rows)})
    return {"valid": len(rows), "rejected": len(rejected),
            "hours": sum(x["seconds"] for x in rows)/3600}

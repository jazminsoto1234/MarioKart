"""Evaluación (§6): matriz de escenarios, una celda a la vez, con IC de Wilson e IQM. Reanudable."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import math

import numpy as np
import torch

from stk_agent.util import atomic_json

from .config import Config, ScenarioClass, config_from_dict
from .model import ActorCritic
from .preprocess import FrameStack, make_batch, resize_frames
from .session import Session
from .vec_env import VecRace


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def iqm(values) -> float:
    """Media del 50 % central."""
    x = np.sort(np.asarray(values, float))
    if len(x) == 0:
        return float("nan")
    q = len(x) // 4
    return float(x[q: len(x) - q].mean()) if len(x) >= 4 else float(x.mean())


def eval_cells(cfg: Config) -> list[dict]:
    e = cfg.eval
    return [dict(tracks=t, difficulty=d, num_karts=n, items=i, start=s)
            for t, d, n, i, s in product(e.track_sets, e.difficulties, e.num_karts, e.items, e.start)]


def cell_name(c: dict) -> str:
    return f"{c['tracks']}_d{c['difficulty']}_n{c['num_karts']}_{'items' if c['items'] else 'noitems'}_{c['start']}"


def load_policy(path, device) -> tuple[ActorCritic, Config]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = ActorCritic(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


def run_cell(model: ActorCritic, cfg: Config, cell: dict, device, session: Session) -> list[dict]:
    """Corre `races_per_cell` carreras con la política determinista."""
    e, o = cfg.eval, cfg.obs
    scenario = ScenarioClass(name=cell_name(cell), num_karts=[cell["num_karts"]],
                             difficulty=[cell["difficulty"]], laps=e.laps, reverse_prob=0.5,
                             random_start_slot=cell["start"] == "random", items=cell["items"],
                             max_seconds=e.max_seconds)
    tracks = cfg.env.train_tracks if cell["tracks"] == "train" else cfg.env.heldout_tracks
    n_envs = min(e.num_envs, e.races_per_cell)
    env = VecRace(cfg, tracks=tracks, num_envs=n_envs, seed=12345, scenarios=[scenario])
    results = []
    try:
        stacker = FrameStack(n_envs, o.frame_stack, o.height, o.width, device)
        obs = env.reset()
        stacker.reset(resize_frames(obs["rgb"], o.height, o.width, device))
        while len(results) < e.races_per_cell and not session.should_stop():
            action = model.act(make_batch(stacker.buf, obs, device), deterministic=True)[0]
            obs, _, term, trunc, infos = env.step(action.cpu().numpy())
            frames = resize_frames(obs["rgb"], o.height, o.width, device)
            stacker.push(frames)
            done = np.flatnonzero(term | trunc)
            if len(done):
                idx = torch.as_tensor(done, device=device)
                stacker.reset(frames[idx], idx)
            results += [info["episode"] for info in infos if "episode" in info]
    finally:
        env.close()
    return results[: e.races_per_cell]


def no_nan(d: dict) -> dict:
    """JSON no admite NaN: se guarda como null."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def cell_metrics(cell: dict, eps: list[dict], laps: int) -> dict:
    n = len(eps)
    wins = sum(e["win"] for e in eps)
    return no_nan({**cell, "races": n, "win_rate": wins / max(n, 1), "win_ci95": wilson(wins, n),
            "finished_rate": float(np.mean([e["finished"] for e in eps])) if n else float("nan"),
            "rank_mean": float(np.mean([e["rank"] for e in eps])) if n else float("nan"),
            "rank_iqm": iqm([e["rank"] for e in eps]),
            "rescues_per_lap": float(np.mean([e["rescues"] for e in eps])) / laps if n else float("nan"),
            "offroad_frac": float(np.mean([e["offroad_frac"] for e in eps])) if n else float("nan"),
            "finish_time_iqm": iqm([e["sim_time"] for e in eps if e["finished"]]),
            "episodes": eps})


def evaluate(checkpoint: str, out_dir: str, session: Session, overrides: Config | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_policy(checkpoint, device)
    if overrides is not None:  # eval/env de la config actual, modelo del checkpoint
        cfg.eval, cfg.env.heldout_tracks = overrides.eval, overrides.env.heldout_tracks
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cell in eval_cells(cfg):
        path = out / f"{cell_name(cell)}.json"
        if path.exists() or session.should_stop():
            continue
        print(f"[eval] {cell_name(cell)} ...")
        eps = run_cell(model, cfg, cell, device, session)
        if len(eps) < cfg.eval.races_per_cell:
            break  # parada a mitad de celda: no se guarda, se repite al reanudar
        m = cell_metrics(cell, eps, cfg.eval.laps)
        atomic_json(path, m)
        print(f"[eval]   victoria {m['win_rate']:.2f} {m['win_ci95']}, terminadas {m['finished_rate']:.2f}")
    summary = [{k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k != "episodes"}
               for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    atomic_json(out / "summary.json", summary)
    print(f"[eval] {len(summary)} celdas en {out / 'summary.json'}")

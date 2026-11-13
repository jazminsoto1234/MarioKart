"""Etapa 0: benchmark de decisiones/s, RAM y temperatura con N simuladores."""
from __future__ import annotations

import copy
import time

import numpy as np

from stk_agent.util import atomic_json

from .config import Config
from .session import gpu_temperature
from .vec_env import VecRace


def process_tree_ram_gb() -> float | None:
    """RAM (RSS) de este proceso y sus hijos. Requiere psutil (opcional)."""
    try:
        import psutil
    except ImportError:
        return None
    me = psutil.Process()
    procs = [me, *me.children(recursive=True)]
    return sum(p.memory_info().rss for p in procs if p.is_running()) / 2**30


def random_actions(rng, n: int) -> np.ndarray:
    a = np.zeros((n, 5), np.float32)
    a[:, 0] = rng.uniform(-0.5, 0.5, n)       # steer
    a[:, 1] = rng.uniform(0.3, 1.0, n)        # lon: casi siempre acelera
    return a


def bench_one(cfg: Config, num_envs: int, width: int, height: int, seconds: float) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.env.render_width, cfg.env.render_height = width, height
    rng = np.random.default_rng(0)
    t0 = time.time()
    env = VecRace(cfg, num_envs=num_envs, seed=0)
    try:
        startup = time.time() - t0
        t0 = time.time()
        env.reset()
        reset = time.time() - t0
        steps, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            env.step(random_actions(rng, num_envs))
            steps += num_envs
        dps = steps / (time.time() - t0)
        ram = process_tree_ram_gb()
    finally:
        env.close()
    return {"num_envs": num_envs, "width": width, "height": height, "decisions_per_s": dps,
            "steps_per_hour_M": dps * 3600 / 1e6, "startup_s": startup, "reset_s": reset,
            "ram_gb": ram, "gpu_temp_c": gpu_temperature()}


def bench(cfg: Config, envs: list[int], resolutions: list[tuple[int, int]], seconds: float, out: str):
    results = []
    for w, h in resolutions:
        for n in envs:
            print(f"[bench] {n} entornos a {w}x{h} ...")
            r = bench_one(cfg, n, w, h, seconds)
            results.append(r)
            ram = f"{r['ram_gb']:.1f} GB" if r["ram_gb"] is not None else "n/d (pip install psutil)"
            print(f"[bench]   {r['decisions_per_s']:.0f} dec/s (~{r['steps_per_hour_M']:.2f} M/h), "
                  f"RAM {ram}, GPU {r['gpu_temp_c']} °C, arranque {r['startup_s']:.1f} s, reset {r['reset_s']:.1f} s")
            atomic_json(out, results)
    return results

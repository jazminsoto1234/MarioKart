"""Recompensa por paso de decisión (informe, sección "Una recompensa de progreso...").

r = r_prog + r_wall + r_off + r_rescue + r_pass + r_time + r_smooth + r_end

La misma función etiqueta los datos humanos (offline) y entrena a PPO (online).
"""
from __future__ import annotations

import numpy as np

from .config import RewardConfig
from .features import RaceSnapshot


def compute_reward(cfg: RewardConfig, before: RaceSnapshot, after: RaceSnapshot, *,
                   steer: float, prev_steer: float, lon: float, rescued: bool,
                   terminated: bool, truncated: bool, driving: bool = True) -> tuple[float, dict]:
    """Devuelve la recompensa y el desglose por término (para logging)."""
    terms = dict.fromkeys(("prog", "wall", "off", "rescue", "pass", "time", "smooth", "end"), 0.0)
    if driving:
        dd = float(np.clip(after.distance - before.distance, -cfg.progress_clip, cfg.progress_clip))
        # Progreso enmascarado fuera de pista y durante el rescate (anti-atajos / anti-teleport)
        if after.on_road and not rescued:
            terms["prog"] = cfg.progress_scale * dd

        v0, v1 = before.speed, after.speed
        braking = lon < 0.0
        if not rescued and not braking and (v0 - v1) > cfg.wall_speed_drop:
            terms["wall"] = -cfg.wall_coef * (v0 / cfg.v_max) ** 2
        if not after.on_road:
            terms["off"] = -cfg.offroad_coef * min(v1 / cfg.v_max, 1.5)
        if rescued:
            terms["rescue"] = -cfg.rescue_penalty

        # Adelantamiento de suma cero (GT Sophy R_ps): gap_i = d_i − d_ego, rivales en ventana
        # Se ignoran saltos de distancia (cruce de salida, rescates): no son adelantamientos
        n = min(len(before.rival_distance), len(after.rival_distance))
        ego_jump = abs(after.distance - before.distance) > cfg.progress_clip
        if n and not rescued and not ego_jump:
            g0 = before.rival_distance[:n] - before.distance
            g1 = after.rival_distance[:n] - after.distance
            steady = np.abs(after.rival_distance[:n] - before.rival_distance[:n]) <= cfg.progress_clip
            inside = (g0 > -cfg.pass_window_behind) & (g0 < cfg.pass_window_ahead) & steady
            terms["pass"] = cfg.pass_coef * float(np.sum((g0 - g1)[inside]))

        terms["time"] = -cfg.time_penalty
        terms["smooth"] = -cfg.smooth_coef * abs(steer - prev_steer)

    if terminated:
        n = max(after.num_karts, 1)
        if n == 1:
            terms["end"] = cfg.finish_bonus
        elif cfg.finish_only_if_first:
            terms["end"] = cfg.finish_bonus if after.rank == 1 else 0.0
        else:
            terms["end"] = cfg.finish_bonus * (n - after.rank) / (n - 1)
    elif truncated:
        terms["end"] = -cfg.finish_bonus / 2
    return float(sum(terms.values())), terms

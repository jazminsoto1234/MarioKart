"""Entorno vectorizado sobre stk_agent.Engine (un proceso de simulador por entorno).

- Lockstep paralelo: en cada tick se envía la acción a todos los simuladores y
  luego se recibe de todos, así N procesos simulan a la vez.
- Solo se renderiza en el último tick de cada decisión (action repeat).
- Escenarios por cuotas fijas: cada entorno pertenece a una ScenarioClass
  (informe: "mezcla de escenarios con cuotas fijas, no etapas que se olvidan").
- Rescue lo dispara una REGLA de atasco, no la política.
- Devuelve observaciones crudas: el redimensionado/frame-stack se hace en torch
  (ppo_rl.preprocess), igual que en el agente de despliegue.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import warnings

import numpy as np

from stk_agent.config import Participant, RaceSettings
from stk_agent.engine import Engine

from .config import Config, ScenarioClass
from .features import (ACTION_DIM, RaceSnapshot, TrackGeometry, action_mask,
                       policy_to_env, privileged_dim, privileged_features, summarize)
from .reward import compute_reward

DRIVING_PHASES = {"GO_PHASE", "MUSIC_PHASE", "RACE_PHASE"}
NEUTRAL = {"steer": 0.0, "acceleration": 0.0, "brake": False, "nitro": False,
           "drift": False, "fire": False, "rescue": False}


def assign_scenarios(scenarios: list[ScenarioClass], num_envs: int) -> list[ScenarioClass]:
    """Reparte entornos por cuota (método de mayores restos)."""
    total = sum(s.quota for s in scenarios)
    raw = [s.quota / total * num_envs for s in scenarios]
    counts = [int(math.floor(r)) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: num_envs - sum(counts)]:
        counts[i] += 1
    out = []
    for s, c in zip(scenarios, counts):
        out += [s] * c
    return out


@dataclass
class EpisodeStats:
    ret: float = 0.0
    length: int = 0
    rescues: int = 0
    offroad_steps: int = 0
    terms: dict = field(default_factory=dict)


class RaceInstance:
    def __init__(self, cfg: Config, scenario: ScenarioClass, tracks: list[str], seed: int):
        self.cfg, self.scenario, self.tracks = cfg, scenario, list(tracks)
        self.rng = np.random.default_rng(seed)
        e = cfg.env
        self.repeat = e.tick_hz // cfg.obs.decision_hz
        if e.tick_hz % cfg.obs.decision_hz:
            raise ValueError("obs.decision_hz debe dividir env.tick_hz")
        self.base_settings = RaceSettings(
            track=self.tracks[0], kart=e.kart, laps=scenario.laps, num_karts=max(scenario.num_karts),
            difficulty=scenario.difficulty[0], seed=int(seed % 2**31), player_slot=0,
            width=e.render_width, height=e.render_height, tick_hz=e.tick_hz,
            policy_hz=cfg.obs.decision_hz, rgb_hz=cfg.obs.decision_hz, quality=e.quality,
            max_seconds=scenario.max_seconds)
        self.engine = Engine(self.base_settings)
        self.settings = self.base_settings
        self.slot = 0
        self.geom: TrackGeometry | None = None
        self.snapshot = None
        self.rgb = None

    # ------------------------------------------------------------------ reset
    def sample_settings(self, options: dict | None = None) -> RaceSettings:
        s = self.scenario
        n = int(self.rng.choice(s.num_karts))
        changes = dict(
            track=str(self.rng.choice(self.tracks)), num_karts=n,
            difficulty=int(self.rng.choice(s.difficulty)), laps=s.laps,
            reverse=bool(self.rng.random() < s.reverse_prob),
            seed=int(self.rng.integers(0, 2**31 - 1)), max_seconds=s.max_seconds)
        changes.update(options or {})
        n = changes["num_karts"]
        self.slot = int(self.rng.integers(0, n)) if s.random_start_slot else n - 1
        changes["player_slot"] = self.slot
        return self.base_settings.updated(**changes)

    def begin_reset(self, options: dict | None = None):
        self.settings = self.sample_settings(options)
        self.engine.send("reset", settings=self.settings,
                         participants=[Participant("agent", self.slot)])

    def end_reset(self):
        start = self.engine.recv(timeout=300)
        self.snapshot = start["snapshot"]
        self.geom = TrackGeometry.from_track(start["track"])
        self.rgb = self.snapshot["rgb"]["agent"]

    def driving(self) -> bool:
        return self.snapshot["state"].get("phase") in DRIVING_PHASES

    def init_episode(self):
        self.summary = summarize(self.snapshot["state"], self.slot)
        self.prev_action = np.zeros(ACTION_DIM, np.float32)
        self.history: deque = deque()
        self.last_rescue_t = -1e9
        self.stats = EpisodeStats(terms={})
        self.items_enabled = self.scenario.items

    # ------------------------------------------------------------------ step
    def send_tick(self, action: dict, render: bool):
        self.engine.send("step", actions={"agent": action}, render=render)

    def recv_tick(self):
        self.snapshot = self.engine.recv()
        if self.snapshot["rgb"]:
            self.rgb = self.snapshot["rgb"]["agent"]

    def tick_done(self) -> tuple[bool, bool]:
        k = next(r for r in self.snapshot["state"]["karts"] if r["id"] == self.slot)
        finished = bool(k["has_finished_race"])
        truncated = not finished and (self.snapshot["sim_time"] >= self.settings.max_seconds
                                      or not self.snapshot["running"])
        return finished, truncated

    def should_rescue(self) -> bool:
        e = self.cfg.env
        t = self.snapshot["sim_time"]
        if not self.driving() or t - self.last_rescue_t < e.rescue_cooldown + e.stuck_seconds:
            return False
        while self.history and t - self.history[0][0] > e.stuck_seconds:
            oldest = self.history.popleft()
            if not self.history or t - self.history[0][0] < e.stuck_seconds:
                self.history.appendleft(oldest)
                break
        if not self.history or t - self.history[0][0] < e.stuck_seconds * 0.95:
            return False
        return (self.summary.distance - self.history[0][1]) < e.stuck_min_progress

    def observation(self):
        priv = privileged_features(self.snapshot["state"], self.slot, self.geom)
        mask = action_mask(self.summary, self.items_enabled)
        return self.rgb, priv, mask, self.prev_action.copy()

    def close(self):
        self.engine.close()


class VecRace:
    """N carreras en paralelo. API estilo gymnasium vector (autoreset al terminar)."""

    def __init__(self, cfg: Config, tracks: list[str] | None = None, num_envs: int | None = None,
                 seed: int = 0, scenarios: list[ScenarioClass] | None = None):
        self.cfg = cfg
        n = num_envs or cfg.ppo.num_envs
        self.scenarios = assign_scenarios(scenarios or cfg.env.scenarios, n)
        tracks = list(tracks or cfg.env.train_tracks)
        self.instances: list[RaceInstance] = []
        try:
            for i, sc in enumerate(self.scenarios):
                self.instances.append(RaceInstance(cfg, sc, tracks, seed * 1000 + i))
            available = set(self.instances[0].engine.assets()["tracks"])
            missing = [t for t in tracks if t not in available]
            if missing:
                warnings.warn(f"Pistas no disponibles (se ignoran): {missing}")
            valid = [t for t in tracks if t in available]
            if not valid:
                raise ValueError(f"Ninguna pista válida. Disponibles: {sorted(available)}")
            for inst in self.instances:
                inst.tracks = valid
        except BaseException:
            self.close()
            raise
        self.tracks = valid
        self.num_envs = n
        self.priv_dim = privileged_dim()
        self.repeat = self.instances[0].repeat

    # ---------------------------------------------------------------- helpers
    def _reset_many(self, idx: list[int], options: dict | None = None):
        for i in idx:
            self.instances[i].begin_reset(options)
        for i in idx:
            self.instances[i].end_reset()
        # Avanzar la cuenta atrás en lockstep (sin render) hasta que se pueda conducir
        pending = [i for i in idx if not self.instances[i].driving()]
        for _ in range(60 * 10):
            if not pending:
                break
            for i in pending:
                self.instances[i].send_tick(NEUTRAL, render=False)
            for i in pending:
                self.instances[i].recv_tick()
            pending = [i for i in pending if not self.instances[i].driving()]
        for i in idx:  # un tick con render para tener el primer frame en fase de carrera
            self.instances[i].send_tick(NEUTRAL, render=True)
        for i in idx:
            self.instances[i].recv_tick()
            self.instances[i].init_episode()

    def _stack(self, idx=None):
        idx = range(self.num_envs) if idx is None else idx
        obs = [self.instances[i].observation() for i in idx]
        return {"rgb": np.stack([o[0] for o in obs]),
                "priv": np.stack([o[1] for o in obs]),
                "mask": np.stack([o[2] for o in obs]),
                "prev_action": np.stack([o[3] for o in obs])}

    # -------------------------------------------------------------------- API
    def reset(self, options: dict | None = None):
        self._reset_many(list(range(self.num_envs)), options)
        return self._stack()

    def step(self, actions: np.ndarray):
        """actions: (N,5) en espacio de política (sin recortar)."""
        actions = np.asarray(actions, np.float32)
        n, cfg = self.num_envs, self.cfg
        rescue = [inst.should_rescue() for inst in self.instances]
        env_actions = [policy_to_env(actions[i], cfg.env.brake_threshold, rescue[i]) for i in range(n)]
        before = [inst.summary for inst in self.instances]
        term = np.zeros(n, bool)
        trunc = np.zeros(n, bool)
        for r in range(self.repeat):
            active = [i for i in range(n) if not (term[i] or trunc[i])]
            if not active:
                break
            render = r == self.repeat - 1
            for i in active:
                a = env_actions[i] if (r == 0 or not rescue[i]) else {**env_actions[i], "rescue": False}
                self.instances[i].send_tick(a, render)
            for i in active:
                self.instances[i].recv_tick()
                term[i], trunc[i] = self.instances[i].tick_done()

        rewards = np.zeros(n, np.float32)
        infos: list[dict] = [{} for _ in range(n)]
        for i, inst in enumerate(self.instances):
            after = summarize(inst.snapshot["state"], inst.slot)
            clipped = np.clip(actions[i], -1, 1)
            r, terms = compute_reward(cfg.reward, before[i], after, steer=float(clipped[0]),
                                      prev_steer=float(inst.prev_action[0]), lon=float(clipped[1]),
                                      rescued=rescue[i], terminated=bool(term[i]),
                                      truncated=bool(trunc[i]), driving=inst.driving())
            rewards[i] = r
            inst.summary = after
            applied = np.array([clipped[0], clipped[1], *(actions[i][2:] > 0.5)], np.float32)
            inst.prev_action = applied
            t = inst.snapshot["sim_time"]
            inst.history.append((t, after.distance))
            if rescue[i]:
                inst.last_rescue_t = t
                inst.history.clear()
                inst.stats.rescues += 1
            s = inst.stats
            s.ret += r
            s.length += 1
            s.offroad_steps += int(not after.on_road)
            for k, v in terms.items():
                s.terms[k] = s.terms.get(k, 0.0) + v
            infos[i] = {"rescued": rescue[i]}

        done = [i for i in range(n) if term[i] or trunc[i]]
        final = self._stack(done) if done else None
        for j, i in enumerate(done):
            inst = self.instances[i]
            infos[i].update(
                final_obs={k: v[j] for k, v in final.items()},
                episode={"return": inst.stats.ret, "length": inst.stats.length,
                         "rank": inst.summary.rank, "num_karts": inst.summary.num_karts,
                         "finished": bool(term[i]), "win": bool(term[i] and inst.summary.rank == 1),
                         "rescues": inst.stats.rescues,
                         "offroad_frac": inst.stats.offroad_steps / max(inst.stats.length, 1),
                         "track": inst.settings.track, "reverse": inst.settings.reverse,
                         "difficulty": inst.settings.difficulty, "scenario": inst.scenario.name,
                         "sim_time": inst.snapshot["sim_time"], "terms": dict(inst.stats.terms)})
        if done:
            self._reset_many(done)
        return self._stack(), rewards, term, trunc, infos

    def close(self):
        for inst in self.instances:
            try:
                inst.close()
            except Exception:
                pass

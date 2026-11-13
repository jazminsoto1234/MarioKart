"""RL interface (capaz lo necesitan)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import warnings
import numpy as np
from .types import Action
from .config import RaceSettings, Participant, validate_participants
from .engine import Engine


@dataclass(frozen=True)
class RewardContext:
    before: dict
    after: dict
    action: Action
    participant: str
    slot: int
    seconds: float
    terminated: bool
    truncated: bool


def kart_state(state, slot):
    for row in state['karts']:
        if row['id'] == slot:
            return row
    raise KeyError(f'No existe el kart {slot} en el snapshot')


class MultiAgentEnv:
    """API paralela por diccionarios; no afirma heredar de PettingZoo."""
    def __init__(self, settings=None, participants=None, reward_fn: Callable | None = None,
                 privileged=False, engine_factory=Engine):
        self.settings = settings or RaceSettings()
        self.participants = participants or [Participant('agent', 0)]
        validate_participants(self.settings, self.participants)
        if any(p.kind != 'external' for p in self.participants):
            raise ValueError('Los participantes de entrenamiento deben ser externos; los slots restantes son bots')
        self.possible_agents = [p.name for p in self.participants]
        self.agents = []
        self.reward_fn = reward_fn
        if reward_fn is None:
            warnings.warn('reward_fn no definido: reward=0.0. La librería no diseña ni aprende tu recompensa.', stacklevel=2)
        self.privileged = privileged
        self.engine = engine_factory(self.settings)
        self.rng = np.random.default_rng(self.settings.seed)
        self.snapshot = None
        self.track = None
        self._done = True

    def _info(self, name, snapshot=None):
        snapshot = self.snapshot if snapshot is None else snapshot
        result = {'reward_defined': self.reward_fn is not None}
        if self.privileged:
            # Nunca se añade a observations.
            result['privileged'] = snapshot['state']
        return result

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        episode_seed = int(seed) if seed is not None else int(self.rng.integers(0, 2**31-1))
        changes = {'seed': episode_seed}
        if options:
            unknown = set(options) - {'track', 'reverse', 'difficulty', 'laps'}
            if unknown:
                raise ValueError(f'Opciones reset no soportadas: {unknown}')
            changes.update(options)
        self.settings = self.settings.updated(**changes)
        start = self.engine.reset(self.participants, self.settings)
        self.snapshot, self.track = start['snapshot'], start['track']
        self.agents = list(self.possible_agents)
        self._done = False
        return ({name: self.snapshot['rgb'][name].copy() for name in self.agents},
                {name: self._info(name) for name in self.agents})

    def step(self, actions):
        if self._done:
            raise RuntimeError('Episodio terminado/no iniciado. Ejecuta reset()')
        active = list(self.agents)
        if set(actions) != set(active):
            raise ValueError(f'Se necesitan acciones de {active}')
        validated = {name: Action.from_any(a) for name, a in actions.items()}
        all_actions = {name: validated.get(name, Action()).as_dict() for name in self.possible_agents}
        rewards = {name: 0.0 for name in active}
        terms = {name: False for name in active}
        truncs = {name: False for name in active}
        individual_steps = {name: 0 for name in active}
        terminal_snapshots = {}
        for _ in range(self.settings.action_repeat):
            before = self.snapshot
            self.snapshot = self.engine.step(all_actions, render=True)
            global_limit = self.snapshot['sim_time'] >= self.settings.max_seconds
            for participant in self.participants:
                name = participant.name
                if name not in active or terms[name] or truncs[name]:
                    continue
                individual_steps[name] += 1
                finished = bool(kart_state(self.snapshot['state'], participant.slot)['has_finished_race'])
                truncated = bool(not finished and (global_limit or not self.snapshot['running']))
                terms[name], truncs[name] = finished, truncated
                if finished or truncated:
                    terminal_snapshots[name] = self.snapshot
                if self.reward_fn is not None:
                    ctx = RewardContext(before['state'], self.snapshot['state'], validated[name],
                                        name, participant.slot, 1/self.settings.tick_hz,
                                        finished, truncated)
                    reward = float(self.reward_fn(ctx))
                    if not math.isfinite(reward):
                        raise ValueError('reward_fn devolvió NaN/Inf')
                    rewards[name] += reward
            if all(terms[name] or truncs[name] for name in active):
                break
        observations = {
            name: terminal_snapshots.get(name, self.snapshot)['rgb'][name].copy()
            for name in active
        }
        infos = {
            name: {
                **self._info(name, terminal_snapshots.get(name, self.snapshot)),
                'substeps': individual_steps[name],
                'seconds': individual_steps[name]/self.settings.tick_hz,
            }
            for name in active
        }
        self.agents = [name for name in active if not terms[name] and not truncs[name]]
        self._done = not self.agents
        return observations, rewards, terms, truncs, infos

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrainingEnv:
    """Single-agent interface against built-in bots."""
    def __init__(self, settings=None, reward_fn=None, privileged=False,
                 engine_factory=Engine, slot=0):
        self.multi = MultiAgentEnv(settings, [Participant('agent', slot)], reward_fn,
                                   privileged, engine_factory)

    @property
    def settings(self):
        return self.multi.settings

    @property
    def track(self):
        """Información privilegiada estática, exclusivamente para entrenamiento."""
        return self.multi.track

    def reset(self, *, seed=None, options=None):
        obs, info = self.multi.reset(seed=seed, options=options)
        return obs['agent'], info['agent']

    def step(self, action):
        obs, rew, term, trunc, info = self.multi.step({'agent': action})
        return obs['agent'], rew['agent'], term['agent'], trunc['agent'], info['agent']

    def close(self):
        self.multi.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

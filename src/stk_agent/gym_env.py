from __future__ import annotations

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError('Instala stk-agent para usar el adaptador Gymnasium') from e

from .types import ACTION_NAMES
from .config import RaceSettings
from .env import TrainingEnv


class STKPixelEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 30}

    def __init__(self, settings=None, reward_fn=None, privileged=False,
                 render_mode=None, **kwargs):
        settings = settings or RaceSettings()
        if render_mode not in (None, 'rgb_array', 'human'):
            raise ValueError('render_mode debe ser None, rgb_array o human')
        self.render_mode = render_mode
        self.base = TrainingEnv(settings, reward_fn, privileged, **kwargs)
        self.observation_space = spaces.Box(0, 255, (settings.height, settings.width, 3), dtype=np.uint8)
        self.action_space = spaces.Dict({
            'steer': spaces.Box(-1.0, 1.0, (1,), np.float32),
            'acceleration': spaces.Box(0.0, 1.0, (1,), np.float32),
            **{name: spaces.Discrete(2) for name in ACTION_NAMES[2:]},
        })
        self.last_rgb = None
        self._ui = None
        self.metadata = {**type(self).metadata, 'render_fps': settings.policy_hz}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        race_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31-1))
        self.last_rgb, info = self.base.reset(seed=race_seed, options=options)
        if self.render_mode == 'human':
            self.render()
        return self.last_rgb, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Acción incompatible con action_space')
        scalars = {key: np.asarray(value).item() for key, value in action.items()}
        obs, reward, terminated, truncated, info = self.base.step(scalars)
        self.last_rgb = obs
        if self.render_mode == 'human':
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.last_rgb is None:
            return None
        if self.render_mode == 'human':
            if self._ui is None:
                from .input import HumanUI
                self._ui = HumanUI(self.base.settings.width, self.base.settings.height, scale=1.5)
            self._ui.show(self.last_rgb, 'SuperTuxKart | RL')
            self._ui.events()
            return None
        return self.last_rgb.copy()

    def close(self):
        if self._ui is not None:
            self._ui.close()
            self._ui = None
        self.base.close()

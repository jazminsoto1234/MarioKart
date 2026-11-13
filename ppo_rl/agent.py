"""Agente de despliegue para `stk_agent evaluation`: solo usa act(rgb)."""
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar ppo_rl

from stk_agent import Action  # noqa: E402
from ppo_rl.config import config_from_dict  # noqa: E402
from ppo_rl.features import ACTION_DIM, policy_to_env  # noqa: E402
from ppo_rl.model import ActorCritic  # noqa: E402
from ppo_rl.preprocess import FrameStack, resize_frames  # noqa: E402


class Agent:
    def __init__(self, config: dict):
        want = config.get("device", "cpu")
        self.device = torch.device(want if want == "cpu" or torch.cuda.is_available() else "cpu")
        ckpt = torch.load(config["checkpoint"], map_location=self.device, weights_only=False)
        self.cfg = config_from_dict(ckpt["config"])
        self.model = ActorCritic(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        hz = self.cfg.obs.decision_hz
        self.decide_every = int(config.get("decide_every", 2))      # 30 Hz → 15 Hz
        self.stuck_steps = int(config.get("stuck_seconds", 3.0) * hz)
        self.stuck_diff = float(config.get("stuck_diff", 2.0))      # diferencia media en niveles de gris
        self.warmup_steps = int(config.get("warmup_seconds", 8.0) * hz)
        self.reset()

    def reset(self):
        self.calls, self.decisions, self.cooldown = 0, 0, 0
        self.stack = None
        self.prev = torch.zeros(1, ACTION_DIM, device=self.device)
        self.action = Action().as_dict()
        self.recent = deque(maxlen=self.stuck_steps)

    def act(self, rgb):
        if self.calls % self.decide_every == 0:
            self.action = self._decide(rgb)
        else:
            self.action = {**self.action, "rescue": False}
        self.calls += 1
        return self.action

    def _decide(self, rgb) -> dict:
        o = self.cfg.obs
        frame = resize_frames(rgb, o.height, o.width, self.device)
        if self.stack is None:
            self.stack = FrameStack(1, o.frame_stack, o.height, o.width, self.device)
            self.stack.reset(frame)
        else:
            self.stack.push(frame)
        a = self.model.act_pixels(self.stack.buf, self.prev)[0].cpu().numpy()
        rescue = self._stuck(frame, float(a[1]))
        self.prev = torch.tensor([[np.clip(a[0], -1, 1), np.clip(a[1], -1, 1), *(a[2:] > 0.5)]],
                                 dtype=torch.float32, device=self.device)
        self.decisions += 1
        return policy_to_env(a, self.cfg.env.brake_threshold, rescue)

    def _stuck(self, frame, lon: float) -> bool:
        """Rescue heurístico: frames casi iguales durante `stuck_seconds` mientras acelera."""
        self.recent.append((frame.float().mean(1), lon > 0.3))
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if self.decisions < self.warmup_steps or len(self.recent) < self.stuck_steps:
            return False
        first = self.recent[0][0]
        still = max(float((f - first).abs().mean()) for f, _ in self.recent) < self.stuck_diff
        if still and all(acc for _, acc in self.recent):
            self.recent.clear()
            self.cooldown = self.stuck_steps
            return True
        return False

    def close(self):
        pass

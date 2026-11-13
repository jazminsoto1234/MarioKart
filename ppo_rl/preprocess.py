"""Resize y frame stack. Lo comparten PPO, el BC offline y agent.py para que vean lo mismo."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_frames(rgb, height: int, width: int, device="cpu") -> torch.Tensor:
    """(N,H,W,3) o (H,W,3) uint8 → (N,3,h,w) uint8 con interpolación `area`."""
    x = torch.as_tensor(np.ascontiguousarray(rgb), device=device)
    if x.ndim == 3:
        x = x[None]
    x = x.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(height, width), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8)


class FrameStack:
    """Stack de los últimos K frames por entorno: (N, K*3, h, w) uint8."""

    def __init__(self, num_envs: int, k: int, height: int, width: int, device="cpu"):
        self.k, self.h, self.w = k, height, width
        self.buf = torch.zeros(num_envs, k * 3, height, width, dtype=torch.uint8, device=device)

    def reset(self, frames: torch.Tensor, idx=None):
        """Rellena todo el stack con el primer frame (episodio nuevo)."""
        idx = slice(None) if idx is None else idx
        self.buf[idx] = frames.repeat(1, self.k, 1, 1)

    def push(self, frames: torch.Tensor) -> torch.Tensor:
        self.buf = torch.cat([self.buf[:, 3:], frames], dim=1)
        return self.buf

    def peek(self, frames: torch.Tensor, idx) -> torch.Tensor:
        """Stack que resultaría de añadir `frames` a los entornos `idx`, sin modificar el buffer."""
        return torch.cat([self.buf[idx, 3:], frames], dim=1)


def make_batch(frames: torch.Tensor, obs: dict, device) -> dict:
    """Junta el stack de frames con el resto de la observación de VecRace en tensores."""
    def t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return {"frames": frames.to(device), "prev_action": t(obs["prev_action"]),
            "priv": t(obs["priv"]), "mask": t(obs["mask"])}

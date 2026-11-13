"""Actor de visión (IMPALA-CNN), crítico asimétrico V(h, s), cabeza auxiliar y normalizadores."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import Config
from .features import ACTION_DIM, ACTION_DIM_CONT, AUX_NAMES, privileged_dim


# ---------------------------------------------------------------- bloques

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaStage(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.res1, self.res2 = ResBlock(c_out), ResBlock(c_out)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 3, stride=2, padding=1)
        return self.res2(self.res1(x))


class ImpalaCNN(nn.Module):
    """uint8 (B, C, h, w) → features (B, out_dim)."""

    def __init__(self, in_ch, channels, out_dim, height, width):
        super().__init__()
        stages, c = [], in_ch
        for ch in channels:
            stages.append(ImpalaStage(c, ch))
            c = ch
        self.stages = nn.Sequential(*stages)
        with torch.no_grad():
            n = self.stages(torch.zeros(1, in_ch, height, width)).numel()
        self.fc = nn.Linear(n, out_dim)

    def forward(self, x):
        x = self.stages(x.float() / 255.0)
        return F.relu(self.fc(F.relu(x).flatten(1)))


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


def init_orthogonal(module, gain=math.sqrt(2)):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class RunningNorm(nn.Module):
    """Media/varianza móviles (Welford por lotes). Se guarda en el state_dict."""

    def __init__(self, dim, clip=10.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / total
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / total) / total
        self.count = total

    @property
    def std(self):
        return (self.var + 1e-8).sqrt()

    def normalize(self, x):
        return ((x - self.mean) / self.std).clamp(-self.clip, self.clip)

    def denormalize(self, x):
        return x * self.std + self.mean


# ---------------------------------------------------------------- distribución híbrida

class HybridDist:
    """Gaussiana 2D (steer, lon) + 3 Bernoulli (drift, fire, nitro) con máscara."""

    def __init__(self, mu, log_std, logits, mask):
        self.mu, self.log_std, self.raw_logits, self.mask = mu, log_std, logits, mask
        self.normal = Normal(mu, log_std.exp().expand_as(mu))
        self.logits = logits.masked_fill(mask < 0.5, -30.0)

    def sample(self):
        cont = self.normal.sample()
        bins = torch.bernoulli(torch.sigmoid(self.logits))
        return torch.cat([cont, bins], -1)

    def mode(self):
        return torch.cat([self.mu, (self.logits > 0).float()], -1)

    def log_prob(self, a):
        lp_c = self.normal.log_prob(a[:, :ACTION_DIM_CONT]).sum(-1)
        b = a[:, ACTION_DIM_CONT:]
        lp_b = -F.binary_cross_entropy_with_logits(self.logits, b, reduction="none")
        return lp_c + (lp_b * self.mask).sum(-1)

    def entropy_heads(self):
        ent_c = self.normal.entropy().sum(-1)
        p = torch.sigmoid(self.logits)
        ent_b = F.binary_cross_entropy_with_logits(self.logits, p, reduction="none")
        return ent_c, (ent_b * self.mask).sum(-1)


def kl_heads(p: HybridDist, q: HybridDist):
    """KL(p‖q) por cabeza: (continua, discreta)."""
    var_p, var_q = (2 * p.log_std).exp(), (2 * q.log_std).exp()
    kl_c = (q.log_std - p.log_std + (var_p + (p.mu - q.mu) ** 2) / (2 * var_q) - 0.5).sum(-1)
    lp, lq = p.raw_logits, q.raw_logits
    kl_b = (torch.sigmoid(lp) * (F.logsigmoid(lp) - F.logsigmoid(lq))
            + torch.sigmoid(-lp) * (F.logsigmoid(-lp) - F.logsigmoid(-lq)))
    return kl_c, (kl_b * p.mask).sum(-1)


# ---------------------------------------------------------------- actor-crítico

class ActorCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m, o = cfg.model, cfg.obs
        self.min_log_std, self.max_log_std = m.min_log_std, m.max_log_std
        self.encoder = ImpalaCNN(3 * o.frame_stack, m.cnn_channels, m.feature_dim, o.height, o.width)
        self.actor = mlp([m.feature_dim + ACTION_DIM, m.trunk_dim, m.trunk_dim])
        self.mu = nn.Linear(m.trunk_dim, ACTION_DIM_CONT)
        self.logits = nn.Linear(m.trunk_dim, ACTION_DIM - ACTION_DIM_CONT)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM_CONT,), m.init_log_std))
        self.aux = nn.Linear(m.feature_dim, len(AUX_NAMES))
        self.critic = mlp([m.feature_dim + ACTION_DIM + privileged_dim(), m.critic_dim, m.critic_dim])
        self.v_head = nn.Linear(m.critic_dim, 1)
        self.priv_norm = RunningNorm(privileged_dim())
        self.ret_norm = RunningNorm(1)

        init_orthogonal(self)
        init_orthogonal(self.mu, 0.01)
        init_orthogonal(self.logits, 0.01)
        init_orthogonal(self.v_head, 1.0)

    # grupos de parámetros (lr distinto por grupo)
    def param_groups(self):
        actor = [*self.actor.parameters(), *self.mu.parameters(), *self.logits.parameters(),
                 self.log_std, *self.aux.parameters()]
        critic = [*self.critic.parameters(), *self.v_head.parameters()]
        return {"encoder": list(self.encoder.parameters()), "actor": actor, "critic": critic}

    def dist(self, feat, prev_action, mask) -> HybridDist:
        h = self.actor(torch.cat([feat, prev_action], -1))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        return HybridDist(self.mu(h), log_std.expand(feat.shape[0], -1), self.logits(h), mask)

    def value(self, feat, prev_action, priv):
        """Valor normalizado. El crítico no propaga gradiente al encoder (stop-grad)."""
        x = torch.cat([feat.detach(), prev_action, self.priv_norm.normalize(priv)], -1)
        return self.v_head(self.critic(x)).squeeze(-1)

    def forward(self, batch: dict):
        feat = self.encoder(batch["frames"])
        dist = self.dist(feat, batch["prev_action"], batch["mask"])
        value = self.value(feat, batch["prev_action"], batch["priv"])
        return dist, value, self.aux(feat)

    @torch.no_grad()
    def act(self, batch: dict, deterministic=False):
        """Devuelve acción (B,5), log-prob y valor DESnormalizado."""
        dist, v, _ = self(batch)
        a = dist.mode() if deterministic else dist.sample()
        return a, dist.log_prob(a), self.ret_norm.denormalize(v[:, None]).squeeze(-1)

    @torch.no_grad()
    def act_pixels(self, frames, prev_action, mask=None):
        """Solo actor (despliegue): sin estado privilegiado."""
        if mask is None:
            mask = torch.ones(frames.shape[0], ACTION_DIM - ACTION_DIM_CONT, device=frames.device)
        return self.dist(self.encoder(frames), prev_action, mask).mode()


def policy_file(model: ActorCritic, cfg: Config) -> dict:
    """Contenido mínimo para agent.py y evaluate.py."""
    return {"model": model.state_dict(), "config": cfg.as_dict()}

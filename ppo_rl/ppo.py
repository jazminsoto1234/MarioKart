"""Etapas 3–4: PPO asimétrico con warmup del crítico, rampa de lr y KL decreciente hacia π_BC."""
from __future__ import annotations

from itertools import cycle
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .features import AUX_SLICE
from .model import ActorCritic, RunningNorm, kl_heads, policy_file
from .preprocess import FrameStack, make_batch, resize_frames
from .session import MetricsLogger, Session, load_checkpoint, save_checkpoint
from .vec_env import VecRace


# ---------------------------------------------------------------- utilidades

class RewardScaler:
    """Divide la recompensa por la desviación del retorno descontado móvil."""

    def __init__(self, num_envs: int, gamma: float):
        self.gamma, self.ret, self.norm = gamma, np.zeros(num_envs), RunningNorm(1)

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + rewards
        self.norm.update(torch.as_tensor(self.ret, dtype=torch.float32)[:, None])
        self.ret[dones] = 0.0
        return (rewards / float(self.norm.std)).astype(np.float32)

    def resize(self, num_envs: int):
        self.ret = np.zeros(num_envs)


def set_learning_rates(opt, cfg: Config, update: int, total_updates: int):
    p = cfg.ppo
    anneal = p.lr_final_frac + (1 - p.lr_final_frac) * max(0.0, 1 - update / max(total_updates, 1))
    since = update - p.critic_warmup_updates
    actor = 0.0 if since < 0 else (min(1.0, (since + 1) / p.actor_lr_ramp_updates) if p.actor_lr_ramp_updates else 1.0)
    encoder = 0.0 if since < p.freeze_encoder_updates else actor
    lrs = {"actor": p.lr_actor * anneal * actor, "encoder": p.lr_actor * anneal * encoder,
           "critic": p.lr_critic * anneal}
    for g in opt.param_groups:
        g["lr"] = lrs[g["name"]]
    return lrs


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def explained_variance(pred, target) -> float:
    var = target.var()
    return float(1 - (target - pred).var() / var) if var > 0 else float("nan")


def summarize_episodes(episodes: list[dict]) -> dict:
    if not episodes:
        return {}
    out = {f"ep/{k}": np.mean([e[k] for e in episodes])
           for k in ("return", "length", "finished", "win", "rescues", "offroad_frac", "rank")}
    for term in episodes[0]["terms"]:
        out[f"terms/{term}"] = np.mean([e["terms"].get(term, 0.0) for e in episodes])
    for sc in {e["scenario"] for e in episodes}:
        group = [e for e in episodes if e["scenario"] == sc]
        out[f"win/{sc}"] = np.mean([e["win"] for e in group])
        out[f"finished/{sc}"] = np.mean([e["finished"] for e in group])
    return out


# ---------------------------------------------------------------- rollout

def collect_rollout(env: VecRace, model: ActorCritic, stacker: FrameStack, obs: dict,
                    scaler: RewardScaler, cfg: Config, device, episodes: list):
    T, N, o = cfg.ppo.rollout_steps, env.num_envs, cfg.obs
    shapes = {"frames": stacker.buf.shape[1:], "prev_action": obs["prev_action"].shape[1:],
              "priv": obs["priv"].shape[1:], "mask": obs["mask"].shape[1:], "actions": (5,)}
    buf = {k: torch.zeros((T, N, *s), device=device, dtype=torch.uint8 if k == "frames" else torch.float32)
           for k, s in shapes.items()}
    for k in ("logp", "values", "rewards", "dones"):
        buf[k] = torch.zeros(T, N, device=device)

    for t in range(T):
        batch = make_batch(stacker.buf, obs, device)
        action, logp, value = model.act(batch)
        for k in ("frames", "prev_action", "priv", "mask"):
            buf[k][t] = batch[k]
        buf["actions"][t], buf["logp"][t], buf["values"][t] = action, logp, value

        obs, reward, term, trunc, infos = env.step(action.cpu().numpy())
        done = term | trunc
        reward = scaler(reward, done)
        frames = resize_frames(obs["rgb"], o.height, o.width, device)

        trunc_idx = np.flatnonzero(trunc)
        if len(trunc_idx):  # bootstrap: r += γ·V(obs_final) si se trunca
            final = {k: np.stack([infos[i]["final_obs"][k] for i in trunc_idx]) for k in ("priv", "mask", "prev_action")}
            final_rgb = np.stack([infos[i]["final_obs"]["rgb"] for i in trunc_idx])
            idx = torch.as_tensor(trunc_idx, device=device)
            final_frames = stacker.peek(resize_frames(final_rgb, o.height, o.width, device), idx)
            v_final = model.act(make_batch(final_frames, final, device))[2]
            reward[trunc_idx] += cfg.ppo.gamma * v_final.cpu().numpy()

        buf["rewards"][t] = torch.as_tensor(reward, device=device)
        buf["dones"][t] = torch.as_tensor(done, dtype=torch.float32, device=device)
        stacker.push(frames)
        if done.any():
            idx = torch.as_tensor(np.flatnonzero(done), device=device)
            stacker.reset(frames[idx], idx)
        episodes += [info["episode"] for info in infos if "episode" in info]

    next_value = model.act(make_batch(stacker.buf, obs, device))[2]
    return buf, obs, next_value


# ---------------------------------------------------------------- actualización

def ppo_update(model, ref, opt, buf, adv, returns, cfg: Config, coefs: dict, bc_batches=None) -> dict:
    p = cfg.ppo
    B = adv.numel()
    flat = {k: v.reshape(B, *v.shape[2:]) for k, v in buf.items()}
    adv, returns = adv.reshape(B), returns.reshape(B)
    model.ret_norm.update(returns[:, None])
    targets = model.ret_norm.normalize(returns[:, None]).squeeze(-1)
    mb_size = B // p.minibatches
    stats, stop = {}, False

    for _ in range(p.epochs):
        perm = torch.randperm(B, device=adv.device)
        for start in range(0, B - mb_size + 1, mb_size):
            idx = perm[start:start + mb_size]
            batch = {k: flat[k][idx] for k in ("frames", "prev_action", "priv", "mask")}
            dist, value, aux = model(batch)
            logratio = dist.log_prob(flat["actions"][idx]) - flat["logp"][idx]
            ratio = logratio.exp()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - p.clip, 1 + p.clip)).mean()
            v_loss = 0.5 * (value - targets[idx]).pow(2).mean()
            ent_c, ent_d = dist.entropy_heads()
            aux_loss = F.mse_loss(aux, batch["priv"][:, AUX_SLICE])
            loss = (pg + p.vf_coef * v_loss + p.aux_coef * aux_loss
                    - p.ent_coef_cont * ent_c.mean() - p.ent_coef_disc * ent_d.mean())

            if ref is not None and (coefs["kl_c"] > 0 or coefs["kl_d"] > 0):
                with torch.no_grad():
                    ref_dist = ref.dist(ref.encoder(batch["frames"]), batch["prev_action"], batch["mask"])
                kl_c, kl_d = kl_heads(dist, ref_dist)
                loss = loss + coefs["kl_c"] * kl_c.mean() + coefs["kl_d"] * kl_d.mean()
                stats.setdefault("kl_ref_cont", []).append(kl_c.mean().item())
                stats.setdefault("kl_ref_disc", []).append(kl_d.mean().item())

            if bc_batches is not None and coefs["bc"] > 0:  # DAPG opcional
                demo = {k: v.to(adv.device) for k, v in next(bc_batches).items()}
                d = model.dist(model.encoder(demo["frames"]), demo["prev_actions"], demo["mask"])
                acts = demo["actions"].clone()
                acts[:, :2] = acts[:, :2].clamp(-0.999, 0.999)
                bc = -(d.log_prob(acts) * (1 - demo["fail"])).mean()
                loss = loss + coefs["bc"] * bc
                stats.setdefault("bc_nll", []).append(bc.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p.max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                for k, v in (("pg_loss", pg.item()), ("v_loss", v_loss.item()), ("aux_loss", aux_loss.item()),
                             ("entropy_cont", ent_c.mean().item()), ("entropy_disc", ent_d.mean().item()),
                             ("approx_kl", approx_kl),
                             ("clipfrac", ((ratio - 1).abs() > p.clip).float().mean().item())):
                    stats.setdefault(k, []).append(v)
            if approx_kl > p.target_kl:
                stop = True
                break
        if stop:
            break

    out = {f"ppo/{k}": float(np.mean(v)) for k, v in stats.items()}
    out["ppo/early_stop"] = float(stop)
    out["ppo/explained_var"] = explained_variance(flat["values"], returns)
    return out


# ---------------------------------------------------------------- entrenamiento

def make_optimizer(model: ActorCritic, cfg: Config):
    groups = [{"params": params, "name": name, "lr": 0.0} for name, params in model.param_groups().items()]
    return torch.optim.Adam(groups, eps=cfg.ppo.adam_eps)


def make_bc_iterator(cfg: Config):
    from .offline import DemoDataset, split_shards
    train_dirs, _ = split_shards(cfg)
    ds = DemoDataset(train_dirs, cfg.obs.frame_stack)
    return cycle(DataLoader(ds, batch_size=cfg.ppo.bc_batch, shuffle=True, drop_last=True, num_workers=1))


def train(cfg: Config, session: Session):
    p, o, run_dir = cfg.ppo, cfg.obs, cfg.run_dir
    device = torch.device(p.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(p.seed)

    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    ref, update, global_step = None, 0, 0
    scaler = RewardScaler(p.num_envs, p.gamma)
    num_envs = p.num_envs

    ckpt = load_checkpoint(run_dir, "last.pt", device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.norm.load_state_dict(ckpt["reward_norm"])
        update, global_step, num_envs = ckpt["update"], ckpt["global_step"], ckpt["num_envs"]
        if ckpt.get("ref") is not None:
            ref = ActorCritic(cfg).to(device)
            ref.load_state_dict(ckpt["ref"])
        print(f"[ppo] Reanudando: update {update}, {global_step:,} pasos")
    elif p.init_checkpoint:
        init = torch.load(p.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        print(f"[ppo] Pesos iniciales: {p.init_checkpoint}")
    if ref is None and p.init_checkpoint and (p.kl_ref_coef_cont > 0 or p.kl_ref_coef_disc > 0):
        ref = copy.deepcopy(model)
    if ref is not None:
        ref.eval().requires_grad_(False)
    bc_batches = make_bc_iterator(cfg) if p.bc_coef > 0 else None

    def make_env(n):
        return VecRace(cfg, num_envs=n, seed=p.seed * 7919 + update)

    def checkpoint():
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "reward_norm": scaler.norm.state_dict(), "update": update, "global_step": global_step,
                 "num_envs": num_envs, "ref": ref.state_dict() if ref is not None else None,
                 "config": cfg.as_dict()}
        save_checkpoint(state, run_dir, cfg.session.keep_checkpoints)
        torch.save(policy_file(model, cfg), run_dir / "policy.pt")

    logger = MetricsLogger(run_dir)
    env = make_env(num_envs)
    try:
        scaler.resize(num_envs)
        stacker = FrameStack(num_envs, o.frame_stack, o.height, o.width, device)
        obs = env.reset()
        stacker.reset(resize_frames(obs["rgb"], o.height, o.width, device))
        episodes: list[dict] = []

        while global_step < p.total_steps and not session.should_stop():
            total_updates = p.total_steps // (p.rollout_steps * num_envs)
            lrs = set_learning_rates(opt, cfg, update, total_updates)
            t0 = time.time()
            model.eval()
            buf, obs, next_value = collect_rollout(env, model, stacker, obs, scaler, cfg, device, episodes)
            sps = buf["rewards"].numel() / (time.time() - t0)
            model.train()

            adv, returns = compute_gae(buf["rewards"], buf["values"], buf["dones"], next_value, p.gamma, p.gae_lambda)
            decay = p.kl_ref_decay ** max(0, update - p.critic_warmup_updates)
            coefs = {"kl_c": p.kl_ref_coef_cont * decay, "kl_d": p.kl_ref_coef_disc * decay,
                     "bc": p.bc_coef * p.bc_decay ** update}
            metrics = ppo_update(model, ref, opt, buf, adv, returns, cfg, coefs, bc_batches)
            update += 1
            global_step += buf["rewards"].numel()

            if update % p.log_every == 0:
                std = model.log_std.detach().clamp(cfg.model.min_log_std, cfg.model.max_log_std).exp()
                metrics.update(summarize_episodes(episodes))
                metrics.update({"perf/sps": sps, "perf/num_envs": num_envs, "lr/actor": lrs["actor"],
                                "lr/critic": lrs["critic"], "std/steer": std[0], "std/lon": std[1],
                                "kl_coef/cont": coefs["kl_c"], "action/lon_mean": buf["actions"][..., 1].mean()})
                logger.log(global_step, metrics, {"action/lon": buf["actions"][..., 1].flatten().cpu()})
                print(f"[ppo] u{update} {global_step:,} pasos | {sps:.0f} sps | "
                      f"ret {metrics.get('ep/return', float('nan')):.2f} | fin {metrics.get('ep/finished', float('nan')):.2f} | "
                      f"kl {metrics.get('ppo/approx_kl', 0):.4f} | ev {metrics['ppo/explained_var']:.2f}")
                episodes.clear()
            if session.time_to_save() or update % p.save_every == 0:
                checkpoint()

            if session.thermal_pause() and session.hot_events >= 2 and num_envs > 2:
                num_envs -= 1
                print(f"[ppo] Calor repetido: bajo a {num_envs} entornos.")
                env.close()
                env = make_env(num_envs)
                scaler.resize(num_envs)
                stacker = FrameStack(num_envs, o.frame_stack, o.height, o.width, device)
                obs = env.reset()
                stacker.reset(resize_frames(obs["rgb"], o.height, o.width, device))
                session.hot_events = 0
        checkpoint()
        print(f"[ppo] Guardado en {run_dir}. Reanudar = mismo comando.")
    finally:
        env.close()
        logger.close()

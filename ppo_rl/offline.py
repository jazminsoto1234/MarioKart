"""Etapas 1 y 2: caché de demostraciones a 15 Hz y preentrenamiento AWR-BC + crítico."""
from __future__ import annotations

from pathlib import Path
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from stk_agent.dataset import Episode, discover_episodes
from stk_agent.util import atomic_json

from .config import Config
from .features import (ACTION_DIM, AUX_SLICE, TrackGeometry, action_mask, env_to_policy,
                       privileged_features, summarize)
from .model import ActorCritic, policy_file
from .preprocess import resize_frames
from .reward import compute_reward
from .session import MetricsLogger, Session, load_checkpoint, save_checkpoint

DRIVING_PHASES = {"GO_PHASE", "MUSIC_PHASE", "RACE_PHASE"}
ARRAYS = ("frames", "actions", "prev_actions", "priv", "mask", "reward", "returns", "fail")


# ================================================================ Etapa 1: caché

def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out, g = np.zeros_like(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def failure_flags(n: int, events: list[int], window: int) -> np.ndarray:
    """1 en los `window` pasos antes de cada rescate/choque (y en el propio paso)."""
    fail = np.zeros(n, np.float32)
    for i in events:
        fail[max(0, i - window): i + 1] = 1.0
    return fail


def episode_to_arrays(ep: Episode, cfg: Config) -> tuple[dict, dict]:
    """Remuestrea un episodio humano a decision_hz y calcula features, máscara y recompensa."""
    o = cfg.obs
    repeat = cfg.env.tick_hz // o.decision_hz
    slot = int(ep.meta["human_slot"])
    geom = TrackGeometry.from_track(ep.load_track())
    cols = {k: [] for k in ("frames", "actions", "prev_actions", "priv", "mask", "reward")}
    events, terms_sum = [], {}
    prev = np.zeros(ACTION_DIM, np.float32)

    for block in ep.iter_action_blocks(repeat=repeat, privileged=True):
        s0, s1 = block["privileged"], block["next_privileged"]
        if s0.get("phase") not in DRIVING_PHASES:
            continue
        a, rescued = env_to_policy(block["action_sequence"])
        before, after = summarize(s0, slot), summarize(s1, slot)
        finished = after.finished and not before.finished
        truncated = bool(block["truncated"]) and not finished
        r, terms = compute_reward(cfg.reward, before, after, steer=float(a[0]),
                                  prev_steer=float(prev[0]), lon=float(a[1]), rescued=rescued,
                                  terminated=finished, truncated=truncated)
        cols["frames"].append(resize_frames(block["observation"], o.height, o.width)[0].numpy())
        cols["actions"].append(a)
        cols["prev_actions"].append(prev)
        cols["priv"].append(privileged_features(s0, slot, geom))
        cols["mask"].append(action_mask(before, items_enabled=True))
        cols["reward"].append(r)
        for k, v in terms.items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        if rescued or terms["wall"] < 0:
            events.append(len(cols["reward"]) - 1)
        prev = np.array([a[0], a[1], *(a[2:] > 0.5)], np.float32)
        if finished or truncated or before.finished:
            break

    arrays = {k: np.stack(v) for k, v in cols.items()}
    arrays["reward"] = arrays["reward"].astype(np.float32)
    arrays["returns"] = discounted_returns(arrays["reward"], cfg.offline.gamma)
    window = int(round(cfg.offline.failure_window_s * o.decision_hz))
    arrays["fail"] = failure_flags(len(arrays["reward"]), events, window)
    s = ep.settings
    meta = {"episode_id": ep.meta["episode_id"], "track": s["track"], "reverse": s["reverse"],
            "difficulty": s["difficulty"], "group": ep.meta.get("group_id"),
            "driver": ep.meta.get("driver_id"), "steps": len(arrays["reward"]),
            "return": float(arrays["reward"].sum()), "terms": terms_sum,
            "fail_frac": float(arrays["fail"].mean())}
    return arrays, meta


def write_shard(arrays: dict, meta: dict, shard_dir: Path):
    """Escritura atómica: carpeta .partial → rename."""
    tmp = shard_dir.with_name(shard_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for k, v in arrays.items():
        np.save(tmp / f"{k}.npy", v)
    atomic_json(tmp / "meta.json", meta)
    os.replace(tmp, shard_dir)


def episode_paths(cfg: Config) -> list[Path]:
    if cfg.offline.split_file:
        split = json.loads(Path(cfg.offline.split_file).read_text(encoding="utf-8"))
        return [Path(p) for key in ("train", "val") for p in split[key]]
    return discover_episodes(cfg.offline.data_root)


def build_cache(cfg: Config, session: Session):
    """Un shard por episodio; los ya hechos se saltan (reanudable)."""
    shards = Path(cfg.offline.cache_dir) / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    paths = episode_paths(cfg)
    for n, path in enumerate(paths, 1):
        if session.should_stop():
            break
        ep = Episode(path)
        out = shards / ep.meta["episode_id"]
        if (out / "meta.json").exists():
            continue
        arrays, meta = episode_to_arrays(ep, cfg)
        write_shard(arrays, meta, out)
        print(f"[cache] {n}/{len(paths)} {meta['track']}: {meta['steps']} pasos, "
              f"retorno {meta['return']:.1f}, fallo {meta['fail_frac']:.1%}")
    index = [json.loads((d / "meta.json").read_text(encoding="utf-8"))
             for d in sorted(shards.iterdir()) if (d / "meta.json").exists()]
    atomic_json(Path(cfg.offline.cache_dir) / "index.json", index)
    returns = np.concatenate([np.load(shards / m["episode_id"] / "returns.npy") for m in index])
    print(f"[cache] {len(index)} episodios, {len(returns)} pasos. "
          f"Retorno MC: media {returns.mean():.2f}, p10 {np.percentile(returns, 10):.2f}, "
          f"p90 {np.percentile(returns, 90):.2f}")


# ================================================================ dataset

class DemoDataset(Dataset):
    """Lee los shards con memmap y arma el frame stack (K frames del mismo episodio)."""

    def __init__(self, shard_dirs: list[Path], frame_stack: int):
        self.k, self.dirs, self._shards = frame_stack, list(shard_dirs), None
        lengths = [len(np.load(d / "reward.npy", mmap_mode="r")) for d in self.dirs]
        self.index = np.array([(i, t) for i, n in enumerate(lengths) for t in range(n)],
                              np.int64).reshape(-1, 2)

    @property
    def shards(self):
        # Se abre en cada worker: pickle de un memmap copiaría todo a RAM
        if self._shards is None:
            self._shards = [{k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ARRAYS} for d in self.dirs]
        return self._shards

    def __getstate__(self):
        return {**self.__dict__, "_shards": None}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, t = self.index[j]
        s = self.shards[i]
        ids = np.clip(np.arange(t - self.k + 1, t + 1), 0, None)
        frames = np.asarray(s["frames"][ids])
        return {"frames": torch.from_numpy(frames.reshape(-1, *frames.shape[2:])),
                **{k: torch.from_numpy(np.array(s[k][t], np.float32))
                   for k in ("actions", "prev_actions", "priv", "mask", "returns", "fail")}}

    def stats(self, key):
        x = np.concatenate([np.asarray(s[key], np.float32).reshape(len(s[key]), -1) for s in self.shards])
        return torch.from_numpy(x.mean(0)), torch.from_numpy(x.var(0)), len(x)


def split_shards(cfg: Config) -> tuple[list[Path], list[Path]]:
    """Train/val por episodio (o por grupo con split.json). Excluye pistas reservadas."""
    root = Path(cfg.offline.cache_dir)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    index = [m for m in index if m["track"] not in set(cfg.env.heldout_tracks)]
    if cfg.offline.split_file:
        split = json.loads(Path(cfg.offline.split_file).read_text(encoding="utf-8"))
        val_ids = {Path(p).name for p in split["val"]}
    else:
        ids = sorted(m["episode_id"] for m in index)
        rng = np.random.default_rng(cfg.offline.seed)
        n_val = max(1, int(round(len(ids) * cfg.offline.val_fraction))) if len(ids) > 1 else 0
        val_ids = set(rng.permutation(ids)[:n_val])
    to_dir = lambda m: root / "shards" / m["episode_id"]
    return ([to_dir(m) for m in index if m["episode_id"] not in val_ids],
            [to_dir(m) for m in index if m["episode_id"] in val_ids])


# ================================================================ Etapa 2: AWR-BC + crítico

def offline_loss(model: ActorCritic, batch: dict, cfg: Config, awr: bool):
    oc = cfg.offline
    feat = model.encoder(batch["frames"])
    dist = model.dist(feat, batch["prev_actions"], batch["mask"])
    v = model.value(feat, batch["prev_actions"], batch["priv"])
    target = model.ret_norm.normalize(batch["returns"][:, None]).squeeze(-1)
    v_loss = 0.5 * (v - target).pow(2).mean()

    adv = (target - v).detach()
    w = torch.ones_like(adv)
    if awr:
        w = torch.exp(oc.awr_beta * adv / (adv.std() + 1e-6)).clamp(max=oc.awr_max_weight)
    w = w * (1.0 - batch["fail"])
    actions = batch["actions"].clone()
    actions[:, :2] = actions[:, :2].clamp(-0.999, 0.999)
    bc = -(w * dist.log_prob(actions)).sum() / w.sum().clamp_min(1.0)

    aux = F.mse_loss(model.aux(feat), batch["priv"][:, AUX_SLICE])
    loss = bc + cfg.ppo.vf_coef * v_loss + oc.aux_coef * aux
    return loss, {"bc_nll": bc.item(), "v_loss": v_loss.item(), "aux": aux.item(),
                  "w_mean": w.mean().item(), "steer_mae": (dist.mu[:, 0] - actions[:, 0]).abs().mean().item()}


@torch.no_grad()
def validate(model, loader, cfg, device) -> dict:
    model.eval()
    totals, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        _, info = offline_loss(model, batch, cfg, awr=False)
        for k, v in info.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}


def pretrain(cfg: Config, session: Session):
    oc, run_dir = cfg.offline, cfg.run_dir
    device = torch.device(oc.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(oc.seed)
    train_dirs, val_dirs = split_shards(cfg)
    train_ds, val_ds = DemoDataset(train_dirs, cfg.obs.frame_stack), DemoDataset(val_dirs, cfg.obs.frame_stack)
    print(f"[pretrain] train {len(train_ds)} pasos ({len(train_dirs)} ep.), val {len(val_ds)} ({len(val_dirs)} ep.)")

    model = ActorCritic(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=oc.lr, eps=1e-5)
    use_amp = oc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epoch, start_batch = 0, 0

    ckpt = load_checkpoint(run_dir, "pretrain_last.pt", device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        epoch, start_batch = ckpt["epoch"], ckpt["batch"]
        torch.set_rng_state(ckpt["torch_rng"])
        print(f"[pretrain] Reanudando en época {epoch}, batch {start_batch}")
    else:
        for norm, key in ((model.priv_norm, "priv"), (model.ret_norm, "returns")):
            mean, var, n = train_ds.stats(key)
            norm.mean.copy_(mean), norm.var.copy_(var), norm.count.fill_(n)

    def state(e, b):
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": e, "batch": b, "torch_rng": torch.get_rng_state(), "config": cfg.as_dict()}

    logger = MetricsLogger(run_dir)
    loader_kw = dict(batch_size=oc.batch_size, num_workers=oc.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=False)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw) if len(val_ds) else None
    steps_per_epoch = len(train_ds) // oc.batch_size
    try:
        while epoch < oc.epochs:
            perm = np.random.default_rng(oc.seed + epoch).permutation(len(train_ds))
            subset = Subset(train_ds, perm[start_batch * oc.batch_size:].tolist())
            loader = DataLoader(subset, shuffle=False, drop_last=True, **loader_kw)
            awr = epoch >= oc.awr_warmup_epochs
            for b, batch in enumerate(loader, start=start_batch):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    loss, info = offline_loss(model, batch, cfg, awr)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                step = epoch * steps_per_epoch + b
                if b % 50 == 0:
                    logger.log(step, {f"train/{k}": v for k, v in info.items()})
                    print(f"[pretrain] ep {epoch} b {b}/{steps_per_epoch} " +
                          " ".join(f"{k}={v:.3f}" for k, v in info.items()))
                if session.time_to_save() or b % oc.save_every_batches == 0:
                    save_checkpoint(state(epoch, b + 1), run_dir, cfg.session.keep_checkpoints, "pretrain_last.pt")
                if session.should_stop():
                    save_checkpoint(state(epoch, b + 1), run_dir, cfg.session.keep_checkpoints, "pretrain_last.pt")
                    return
                session.thermal_pause()
            epoch, start_batch = epoch + 1, 0
            if val_loader:
                metrics = validate(model, val_loader, cfg, device)
                logger.log(epoch * steps_per_epoch, metrics)
                print(f"[pretrain] fin época {epoch}: " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            save_checkpoint(state(epoch, 0), run_dir, cfg.session.keep_checkpoints, "pretrain_last.pt")
            torch.save(policy_file(model, cfg), run_dir / "bc.pt")
        print(f"[pretrain] Terminado. Pesos para PPO: {run_dir / 'bc.pt'}")
    finally:
        logger.close()

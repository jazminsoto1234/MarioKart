"""Configuración del pipeline. Todo se carga desde YAML/JSON a dataclasses.

Los valores por defecto siguen la tabla de hiperparámetros del informe
(reports/Entrenamiento PPO autónomo en SuperTuxKart.md). Los pesos de
recompensa son puntos de partida razonados, no valores publicados.
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import json
from typing import Any

import yaml


@dataclass
class ObsConfig:
    height: int = 72                  # 16:9 a baja resolución; IMPALA acepta cualquier tamaño
    width: int = 128
    frame_stack: int = 4
    decision_hz: int = 15             # tick_hz=60 → action repeat 4


@dataclass
class RewardConfig:
    progress_scale: float = 0.1       # por metro de avance sobre la línea central
    progress_clip: float = 8.0        # m por decisión (elimina saltos por vuelta/teleport)
    v_max: float = 30.0               # m/s, normaliza penalizaciones
    wall_coef: float = 2.0            # −c·(v/vmax)² en choque (proxy: caída brusca de velocidad)
    wall_speed_drop: float = 4.0      # m/s perdidos en una decisión sin frenar = choque
    offroad_coef: float = 0.5         # −c·(v/vmax) fuera de pista
    rescue_penalty: float = 4.0
    pass_coef: float = 0.05           # bonus de adelantamiento de suma cero (por metro de gap)
    pass_window_behind: float = 20.0
    pass_window_ahead: float = 40.0
    time_penalty: float = 0.01
    smooth_coef: float = 0.02         # −c·|Δsteer|
    finish_bonus: float = 10.0        # +B·(N−rank)/(N−1); −B/2 si timeout
    finish_only_if_first: bool = False


@dataclass
class ScenarioClass:
    """Una clase de escenario con su cuota fija de entornos paralelos."""
    name: str = "time_trial"
    quota: float = 1.0
    num_karts: list[int] = field(default_factory=lambda: [1])
    difficulty: list[int] = field(default_factory=lambda: [0])
    laps: int = 1
    reverse_prob: float = 0.5
    random_start_slot: bool = True
    items: bool = False               # False → fire y nitro enmascarados
    max_seconds: float = 180.0


@dataclass
class EnvConfig:
    render_width: int = 640           # igual que la evaluación/datos humanos: el HUD se ve igual
    render_height: int = 360
    quality: str = "sd"
    tick_hz: int = 60
    train_tracks: list[str] = field(default_factory=lambda: [
        "lighthouse", "snowmountain", "minigolf", "xr591", "snowtuxpeak",
        "black_forest", "fortmagma", "olivermath", "cocoa_temple", "zengarden",
        "hacienda", "scotland", "volcano_island", "candela_city", "abyss",
    ])
    heldout_tracks: list[str] = field(default_factory=lambda: [
        "ravenbridge_mansion", "sandtrack", "stk_enterprise", "mines",
    ])
    kart: str = "random"
    scenarios: list[ScenarioClass] = field(default_factory=lambda: [ScenarioClass()])
    # Rescue por regla (no lo elige la política)
    stuck_seconds: float = 2.5
    stuck_min_progress: float = 1.5   # m en la ventana
    rescue_cooldown: float = 3.0
    brake_threshold: float = -0.15    # lon < umbral → freno


@dataclass
class ModelConfig:
    cnn_channels: list[int] = field(default_factory=lambda: [16, 32, 32])
    feature_dim: int = 256
    trunk_dim: int = 256
    critic_dim: int = 512
    init_log_std: float = -0.69       # σ≈0.5
    min_log_std: float = -2.5
    max_log_std: float = 0.5


@dataclass
class PPOConfig:
    num_envs: int = 4                 # ≤ 6 en la laptop (§10)
    rollout_steps: int = 512
    total_steps: int = 10_000_000
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.15
    lr_actor: float = 2.5e-4
    lr_critic: float = 2.5e-4
    lr_final_frac: float = 0.1        # recocido lineal hasta lr·frac
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef_cont: float = 1e-3
    ent_coef_disc: float = 0.01
    aux_coef: float = 0.1             # predicción auxiliar de estado privilegiado
    target_kl: float = 0.03           # corta épocas si approx_kl lo supera
    norm_reward: bool = True
    norm_value: bool = True
    # Arranque desde BC (PIRLNav / VPT / DAPG)
    init_checkpoint: str | None = None
    critic_warmup_updates: int = 0    # actor congelado, solo aprende el crítico
    actor_lr_ramp_updates: int = 0    # rampa lineal 0 → lr_actor
    freeze_encoder_updates: int = 0
    kl_ref_coef_cont: float = 0.0     # KL(π‖π_BC) por cabeza
    kl_ref_coef_disc: float = 0.0
    kl_ref_decay: float = 0.995       # multiplicativo por update
    bc_coef: float = 0.0              # término DAPG opcional sobre datos humanos
    bc_decay: float = 0.99
    bc_batch: int = 128
    seed: int = 1
    device: str = "cuda"
    log_every: int = 1
    save_every: int = 20
    eval_every: int = 0


@dataclass
class OfflineConfig:
    data_root: str = "data/local"
    cache_dir: str = "ppo_rl/cache"
    split_file: str = ""              # split.json de `ingest split`; vacío = split por episodio
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    awr_beta: float = 2.0             # w = clip(exp(β·Â/σ_A), 0, w_max)
    awr_max_weight: float = 20.0
    awr_warmup_epochs: int = 3        # primeras épocas: BC uniforme mientras el crítico converge
    failure_window_s: float = 1.5     # tramo antes de un rescate/choque: peso ≈ 0
    gamma: float = 0.99
    aux_coef: float = 0.1
    val_fraction: float = 0.1
    device: str = "cuda"
    seed: int = 0
    num_workers: int = 2
    amp: bool = True
    save_every_batches: int = 500


@dataclass
class SessionConfig:
    max_hours: float = 0.0            # 0 = sin límite
    until: str = ""                   # "HH:MM" hora local de parada
    checkpoint_minutes: float = 15.0
    keep_checkpoints: int = 3
    torch_threads: int = 4
    low_priority: bool = True
    prevent_sleep: bool = True
    gpu_temp_pause: float = 85.0      # °C: pausa
    gpu_temp_resume: float = 75.0     # °C: reanuda
    temp_check_seconds: float = 60.0


@dataclass
class EvalConfig:
    races_per_cell: int = 50
    difficulties: list[int] = field(default_factory=lambda: [0, 1, 2])
    num_karts: list[int] = field(default_factory=lambda: [4, 8])     # 3 y 7 rivales
    items: list[bool] = field(default_factory=lambda: [False, True])
    start: list[str] = field(default_factory=lambda: ["random", "last"])
    track_sets: list[str] = field(default_factory=lambda: ["train", "heldout"])
    laps: int = 1
    max_seconds: float = 240.0
    num_envs: int = 4


@dataclass
class Config:
    run_name: str = "ppo_stk"
    out_dir: str = "runs/ppo_rl"
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name


def _build(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Claves desconocidas en {cls.__name__}: {sorted(unknown)}")
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if is_dataclass(default) and isinstance(value, dict):
            value = _build(type(default), value)
        elif name == "scenarios":
            value = [_build(ScenarioClass, v) for v in value]
        kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict) -> Config:
    return _build(Config, data)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Carga YAML/JSON y aplica overrides tipo `ppo.num_envs=4`."""
    data: dict = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if str(path).endswith(".json") else (yaml.safe_load(text) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override inválido: {item!r} (usa sección.clave=valor)")
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return _build(Config, data)

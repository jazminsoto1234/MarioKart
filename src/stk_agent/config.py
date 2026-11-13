from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math

RANDOM = 'random'


@dataclass(frozen=True)
class RaceSettings:
    track: str = 'lighthouse'
    kart: str = 'random'              # Identifier returned by assets(), or 'random'.
    bot_kart: str = 'random'          # Identifier, 'random', or 'same_as_player'.
    kart_pool: tuple[str, ...] | None = None
    avoid_duplicate_karts: bool = True
    show_names: bool = False
    laps: int = 3
    num_karts: int = 8
    difficulty: int = 2
    seed: int | str = 42               # collect configs may use 'random'; eval/doctor should use an int.
    player_slot: int | str = RANDOM    # Collection-only hint; explicit eval participants still choose their slots.
    reverse: bool = False
    width: int = 640
    height: int = 360
    tick_hz: int = 60                # API stepping / human display target.
    policy_hz: int = 30
    rgb_hz: int = 30                 # Stored video; display remains tick_hz.
    quality: str = 'sd'              # sd is the default; hd/ld remain available.
    overlay: bool = True
    max_seconds: float = 600.0

    def __post_init__(self):
        for field in ('laps','num_karts','width','height','tick_hz','policy_hz','rgb_hz'):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{field} debe ser un entero positivo')
        if self.num_karts > 20:
            raise ValueError('Esta librería limita num_karts a 20; no es el límite del motor')
        if self.width % 2 or self.height % 2:
            raise ValueError('Ancho y alto deben ser pares')
        if 120 % self.tick_hz:
            raise ValueError('tick_hz debe dividir 120; no cambia la física nativa')
        for name in ('policy_hz','rgb_hz'):
            if self.tick_hz % getattr(self, name):
                raise ValueError(f'{name} debe dividir tick_hz')
        if type(self.difficulty) is not int or self.difficulty not in (0,1,2):
            raise ValueError('difficulty debe ser 0, 1 o 2')
        if self.seed != RANDOM and (type(self.seed) is not int or not 0 <= self.seed < 2**31):
            raise ValueError("seed debe ser un entero en [0, 2**31) o 'random'")
        if self.player_slot != RANDOM:
            if type(self.player_slot) is not int or not 0 <= self.player_slot < self.num_karts:
                raise ValueError("player_slot debe ser un slot válido o 'random'")
        if self.quality not in ('ld','sd','hd'):
            raise ValueError('quality debe ser ld, sd o hd')
        for name in ('overlay','reverse','avoid_duplicate_karts','show_names'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} debe ser bool')
        if not self.overlay:
            raise ValueError('El contrato visual requiere overlay=true (HUD)')
        if (not isinstance(self.max_seconds, (float, int)) or isinstance(self.max_seconds, bool)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError('max_seconds debe ser positivo y finito')
        for name in ('track','kart','bot_kart'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} debe ser un texto no vacío')
        if self.kart_pool is not None:
            if (isinstance(self.kart_pool, str) or not self.kart_pool
                    or any(not isinstance(k, str) or not k or k in ('random','same_as_player')
                           for k in self.kart_pool)):
                raise ValueError('kart_pool debe ser una lista no vacía de identificadores concretos')
            if len(set(self.kart_pool)) != len(self.kart_pool):
                raise ValueError('kart_pool no debe contener duplicados')
            object.__setattr__(self, 'kart_pool', tuple(self.kart_pool))

    @property
    def action_repeat(self):
        return self.tick_hz // self.policy_hz

    def wants_rgb(self, tick):
        return tick % (self.tick_hz // self.rgb_hz) == 0

    @property
    def resolved_seed(self):
        if self.seed == RANDOM:
            raise ValueError("La semilla 'random' debe resolverse antes de iniciar el motor")
        return int(self.seed)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        removed = {
            'depth_hz', 'record_depth', 'depth_storage',
            'instance_hz', 'record_instance', 'auxiliary_hz',
        }
        old = sorted(removed.intersection(data))
        if old:
            raise ValueError(
                'La configuración contiene campos retirados: '
                + ', '.join(old)
                + '. Depth e instance segmentation ya no forman parte del proyecto; elimínalos del JSON.'
            )
        return cls(**data)

    def as_dict(self):
        return asdict(self)

    def updated(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class Participant:
    """Slot configurable. kind='bot' invoca IA interna de STK, no un modelo aprendido."""
    name: str
    slot: int
    kind: str = 'external'
    observe: bool = True
    kart: str | None = None

    def __post_init__(self):
        if self.kind not in ('external','bot'):
            raise ValueError('kind debe ser external o bot')
        if type(self.slot) is not int or self.slot < 0 or not self.name:
            raise ValueError('Nombre y slot inválidos')
        if self.kind == 'external' and not self.observe:
            raise ValueError('Un agente externo necesita su cámara')


def validate_participants(settings, participants):
    if not participants or not any(p.observe for p in participants):
        raise ValueError('Se requiere al menos un participante observado')
    if len({p.slot for p in participants}) != len(participants) or len({p.name for p in participants}) != len(participants):
        raise ValueError('Slots/nombres repetidos')
    if any(p.slot >= settings.num_karts for p in participants):
        raise ValueError('Un slot excede num_karts')
    if sum(p.observe for p in participants) > 4:
        raise ValueError('Esta versión limita a cuatro cámaras independientes')

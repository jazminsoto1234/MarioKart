"""Features a partir del estado privilegiado (WorldState) y la geometría estática de la pista.

Se usa idéntico en el entorno online y en el etiquetado de los datos humanos,
para que el crítico, la recompensa y el preentrenamiento vean lo mismo.

Convenciones del motor: X derecha, Y arriba, Z adelante. Se trabaja en el plano XZ.

Espacio de acción de la POLÍTICA (5 dims):
    [steer ∈ [-1,1], lon ∈ [-1,1] (gas>0 / freno<0), drift, fire, nitro]
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ACTION_DIM_CONT = 2
ACTION_DIM_BIN = 3
ACTION_DIM = ACTION_DIM_CONT + ACTION_DIM_BIN
BIN_NAMES = ("drift", "fire", "nitro")

LOOKAHEAD_M = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0)
NUM_RIVALS = 4
NUM_ITEMS = 4
NUM_POWERUP_TYPES = 12
NUM_ATTACHMENT_TYPES = 10

# Índices (dentro del vector privilegiado) que la cabeza auxiliar del actor
# intenta predecir desde píxeles. Son cosas que un jugador SÍ puede inferir
# de la imagen: velocidad, posición lateral, rumbo, curva próxima, ítem.
AUX_NAMES = ("v_fwd", "v_lat", "lateral", "heading_sin", "heading_cos",
             "look20_lat", "look45_lat", "on_road", "energy", "has_powerup")


def _norm2(v):
    n = float(np.hypot(v[0], v[1]))
    return (v / n, n) if n > 1e-6 else (np.array([0.0, 1.0]), 0.0)


@dataclass
class TrackGeometry:
    starts: np.ndarray      # (N,2) xz inicio de cada segmento
    ends: np.ndarray        # (N,2)
    d0: np.ndarray          # (N,) distancia acumulada al inicio
    d1: np.ndarray          # (N,)
    width: np.ndarray       # (N,)
    length: float
    order: np.ndarray       # índices ordenados por d0

    @classmethod
    def from_track(cls, track: dict) -> "TrackGeometry":
        nodes = np.asarray(track["path_nodes"], np.float64)
        dist = np.asarray(track["path_distance"], np.float64)
        starts, ends = nodes[:, 0, [0, 2]], nodes[:, 1, [0, 2]]
        order = np.argsort(dist[:, 0], kind="stable")
        return cls(starts, ends, dist[:, 0], dist[:, 1],
                   np.asarray(track["path_width"], np.float64).reshape(-1),
                   float(np.asarray(track["length"])), order)

    def point_at(self, d: float):
        """Punto central (xz), dirección unitaria y ancho a distancia d a lo largo de la pista."""
        d = d % self.length if self.length > 0 else d
        d0s = self.d0[self.order]
        k = int(np.searchsorted(d0s, d, side="right")) - 1
        i = int(self.order[max(k, 0)])
        seg = self.ends[i] - self.starts[i]
        span = max(self.d1[i] - self.d0[i], 1e-6)
        t = float(np.clip((d - self.d0[i]) / span, 0.0, 1.0))
        direction, _ = _norm2(seg)
        return self.starts[i] + t * seg, direction, float(self.width[i])


def find_kart(state: dict, slot: int) -> dict:
    for k in state["karts"]:
        if k["id"] == slot:
            return k
    raise KeyError(f"slot {slot} no está en el estado")


def kart_frame(k: dict):
    loc = np.asarray(k["location"], np.float64)[[0, 2]]
    front = np.asarray(k["front"], np.float64)[[0, 2]]
    fwd, _ = _norm2(front - loc)
    right = np.array([fwd[1], -fwd[0]])
    return loc, fwd, right


def to_local(p, loc, fwd, right):
    d = np.asarray(p, np.float64) - loc
    return float(d @ right), float(d @ fwd)


@dataclass
class RaceSnapshot:
    """Resumen mínimo para la recompensa (un paso de decisión)."""
    distance: float          # overall_distance (acumula vueltas)
    speed: float
    on_road: bool
    rank: int
    num_karts: int
    finished: bool
    rival_distance: np.ndarray   # overall_distance de los rivales
    energy: float
    has_powerup: bool


def summarize(state: dict, slot: int) -> RaceSnapshot:
    k = find_kart(state, slot)
    vel = np.asarray(k["velocity"], np.float64)
    rivals = np.array([float(r["overall_distance"]) for r in state["karts"] if r["id"] != slot])
    pu = k.get("powerup") or {}
    return RaceSnapshot(
        distance=float(k["overall_distance"]),
        speed=float(np.hypot(vel[0], vel[2])),
        on_road=bool(k.get("is_on_road", True)),
        rank=int(k.get("position") or 1),
        num_karts=len(state["karts"]),
        finished=bool(k.get("has_finished_race")),
        rival_distance=rivals,
        energy=float(k.get("energy") or 0.0),
        has_powerup=int(pu.get("type") or 0) != 0 and int(pu.get("num") or 0) > 0,
    )


def _onehot(i, n):
    v = np.zeros(n, np.float32)
    v[int(np.clip(i, 0, n - 1))] = 1.0
    return v


def privileged_features(state: dict, slot: int, geom: TrackGeometry) -> np.ndarray:
    """Vector privilegiado para el CRÍTICO (y objetivos auxiliares). Nunca entra al actor."""
    k = find_kart(state, slot)
    loc, fwd, right = kart_frame(k)
    vel = np.asarray(k["velocity"], np.float64)[[0, 2]]
    v_fwd, v_lat = float(vel @ fwd), float(vel @ right)
    d_track = float(k["distance_down_track"])

    c, tdir, width = geom.point_at(d_track)
    tright = np.array([tdir[1], -tdir[0]])
    lateral = float((loc - c) @ tright) / max(width / 2, 1.0)
    heading_sin = float(fwd[0] * tdir[1] - fwd[1] * tdir[0])
    heading_cos = float(fwd @ tdir)

    look = []
    for dist in LOOKAHEAD_M:
        p, _, w = geom.point_at(d_track + dist)
        lat, fw = to_local(p, loc, fwd, right)
        look += [lat / 20.0, fw / 80.0, w / 20.0]

    n = len(state["karts"])
    rank = int(k.get("position") or 1)
    lap_frac = (d_track / geom.length) if geom.length > 0 else 0.0

    # Rivales más cercanos por distancia sobre la pista
    me = float(k["overall_distance"])
    rivals = [r for r in state["karts"] if r["id"] != slot]
    rivals.sort(key=lambda r: abs(float(r["overall_distance"]) - me))
    riv = []
    for r in rivals[:NUM_RIVALS]:
        rl = np.asarray(r["location"], np.float64)[[0, 2]]
        lat, fw = to_local(rl, loc, fwd, right)
        rv = np.asarray(r["velocity"], np.float64)[[0, 2]]
        riv += [float(np.clip((float(r["overall_distance"]) - me) / 50.0, -3, 3)),
                float(np.clip(lat / 20.0, -3, 3)), float(np.clip(fw / 50.0, -3, 3)),
                float((rv - vel) @ fwd) / 30.0, 1.0]
    riv += [0.0] * (5 * NUM_RIVALS - len(riv))

    items = []
    for it in state.get("items", []):
        lat, fw = to_local(np.asarray(it["location"], np.float64)[[0, 2]], loc, fwd, right)
        if 0.0 < fw < 60.0 and abs(lat) < 25.0:
            items.append((fw, lat, float(it.get("type") or 0)))
    items.sort()
    itf = []
    for fw, lat, typ in items[:NUM_ITEMS]:
        itf += [lat / 20.0, fw / 60.0, typ / 10.0, 1.0]
    itf += [0.0] * (4 * NUM_ITEMS - len(itf))

    pu = k.get("powerup") or {}
    att = k.get("attachment") or {}
    energy = float(k.get("energy") or 0.0)
    has_pu = float(int(pu.get("type") or 0) != 0 and int(pu.get("num") or 0) > 0)

    aux = [v_fwd / 30.0, v_lat / 10.0, float(np.clip(lateral, -3, 3)), heading_sin, heading_cos,
           look[2 * 3 + 0], look[4 * 3 + 0], float(bool(k.get("is_on_road", True))),
           min(energy / 10.0, 2.0), has_pu]
    base = [rank / max(n, 1), n / 8.0, lap_frac, float(k.get("finished_laps") or 0) / 3.0,
            float(np.clip(k.get("angular_velocity", [0, 0, 0])[1], -5, 5)) / 5.0,
            float(bool(k.get("jumping"))), min(float(k.get("shield_time") or 0.0), 5.0) / 5.0,
            min(int(pu.get("num") or 0), 3) / 3.0, min(float(att.get("time_left") or 0.0), 10.0) / 10.0]
    vec = np.concatenate([
        np.asarray(aux, np.float32), np.asarray(base, np.float32), np.asarray(look, np.float32),
        np.asarray(riv, np.float32), np.asarray(itf, np.float32),
        _onehot(int(pu.get("type") or 0), NUM_POWERUP_TYPES),
        _onehot(int(att.get("type") or 0), NUM_ATTACHMENT_TYPES),
    ]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def privileged_dim() -> int:
    return (len(AUX_NAMES) + 9 + 3 * len(LOOKAHEAD_M) + 5 * NUM_RIVALS + 4 * NUM_ITEMS
            + NUM_POWERUP_TYPES + NUM_ATTACHMENT_TYPES)


AUX_SLICE = slice(0, len(AUX_NAMES))


def action_mask(snap: RaceSnapshot, items_enabled: bool) -> np.ndarray:
    """Máscara de cabezas binarias [drift, fire, nitro]. 1 = permitido."""
    return np.array([1.0,
                     float(items_enabled and snap.has_powerup),
                     float(items_enabled and snap.energy > 0.0)], np.float32)


def policy_to_env(a: np.ndarray, brake_threshold: float = -0.15, rescue: bool = False) -> dict:
    """Acción de política (5) → acción del motor (7 campos de stk_agent.Action)."""
    steer = float(np.clip(a[0], -1, 1))
    lon = float(np.clip(a[1], -1, 1))
    return {"steer": steer,
            "acceleration": max(lon, 0.0),
            "brake": bool(lon < brake_threshold),
            "nitro": bool(a[4] > 0.5),
            "drift": bool(a[2] > 0.5),
            "fire": bool(a[3] > 0.5),
            "rescue": bool(rescue)}


def env_to_policy(seq: np.ndarray) -> tuple[np.ndarray, bool]:
    """Bloque de acciones humanas (T,7) [steer, acc, brake, nitro, drift, fire, rescue]
    → acción de política (5) y si hubo rescate en el bloque."""
    seq = np.asarray(seq, np.float32).reshape(-1, 7)
    steer = float(seq[:, 0].mean())
    lon = float(np.clip(seq[:, 1].mean() - seq[:, 2].mean(), -1, 1))
    drift = float(seq[:, 4].max() > 0.5)
    fire = float(seq[:, 5].max() > 0.5)
    nitro = float(seq[:, 3].max() > 0.5)
    return np.array([steer, lon, drift, fire, nitro], np.float32), bool(seq[:, 6].max() > 0.5)

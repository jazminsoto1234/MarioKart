from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
from dataclasses import asdict
import numpy as np
from .config import RaceSettings, Participant, validate_participants
from .types import Action, check_rgb

ENGINE_VERSION = "0.7.4"
ENGINE_COMMIT = "dd70f6823f248ae1df2ce513839a9b2c8c940c39"
KART_FIELDS = (
    "id", "player_id", "name", "kart_type", "location", "rotation", "front",
    "velocity", "velocity_lc", "angular_velocity", "speed", "size", "shield_time",
    "jumping", "position", "energy", "finished_laps", "lap_time", "finish_time",
    "overall_distance", "distance_down_track", "max_steer_angle", "wheel_base",
    "skeed_factor", "has_finished_race", "race_result", "node", "is_on_road", "lives",
)


def scalar_or_copy(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, (list, tuple)):
        return np.asarray(v).copy()
    if hasattr(v, "value"):
        return int(v.value)
    return v


def extract_world(world):
    """Copia de auditoría: no es una observación válida para el actor."""
    karts = []
    for k in world.karts:
        row = {field: scalar_or_copy(getattr(k, field)) if hasattr(k, field) else None
               for field in KART_FIELDS}
        row["powerup"] = {"type": scalar_or_copy(k.powerup.type), "num": int(k.powerup.num)}
        row["attachment"] = {"type": scalar_or_copy(k.attachment.type),
                             "time_left": float(k.attachment.time_left)}
        karts.append(row)
    items = [{key: scalar_or_copy(getattr(i, key)) for key in ("id", "location", "size", "type")}
             for i in world.items]
    return {"time": float(world.time), "aux_ticks": int(getattr(world, "aux_ticks", 0)),
            "phase": getattr(world.phase, "name", str(world.phase)),
            "karts": karts, "items": items}


class PySTKBackend:
    def __init__(self, settings: RaceSettings):
        from .runtime import register_dll_directory, bootstrap_windows_runtime
        self._dll_directory_handles = register_dll_directory()

        try:
            import pystk2
        except ModuleNotFoundError as e:
            if e.name == "pystk2":
                raise RuntimeError('Falta PySTK2. Reinstala stk-agent.') from e
            raise
        except ImportError as first_error:
            if os.name != "nt" or os.environ.get("STK_DLL_DIR"):
                raise RuntimeError(
                    f"PySTK2 está instalado pero {platform.system()} no pudo cargar sus bibliotecas nativas: {first_error}"
                ) from first_error
            try:
                bootstrap_windows_runtime()
                self._dll_directory_handles += register_dll_directory()
                import sys
                sys.modules.pop("pystk2", None)
                importlib.invalidate_caches()
                pystk2 = importlib.import_module("pystk2")
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo preparar el runtime nativo de PySTK2 en Windows: {e}"
                ) from e
        version = importlib.metadata.version("PySuperTuxKart2")
        if version != ENGINE_VERSION:
            raise RuntimeError(f"Motor {version}; este protocolo requiere PySuperTuxKart2=={ENGINE_VERSION}")
        if hasattr(pystk2, "has_graphics") and not pystk2.has_graphics:
            raise RuntimeError("Se requiere un build con gráficos; un actor RGB no funciona en SERVER_ONLY")
        self.p = pystk2
        self.settings = settings
        self.race = None
        self.world = None
        self.participants = []
        self.frame_id = 0
        self.running = False
        graphics = getattr(pystk2.GraphicsConfig, settings.quality)()
        graphics.screen_width, graphics.screen_height = settings.width, settings.height
        if not hasattr(graphics, "display"):
            raise RuntimeError("El binding instalado no soporta display=False")
        graphics.render, graphics.display = True, False
        # Mantiene una imagen estable para el actor; HUD permanece.
        graphics.motionblur = False
        graphics.mlaa = False
        graphics.dof = False
        pystk2.init(graphics)
        self.provenance = {
            "distribution": "PySuperTuxKart2", "distribution_version": version,
            "reference_commit": ENGINE_COMMIT, "native_version": str(pystk2.__version__),
            "python": platform.python_version(), "platform": platform.platform(),
            "render": True, "display": False, "motionblur": False, "mlaa": False,
            "coordinates": "X right, Y up, Z forward; rotation read is world-to-kart [w,x,y,z]",
            "item_availability": "item list is not a guaranteed available-pickup/event list",
            "bootstrap_steps": 1,
            "quality": settings.quality,
            "bot_implementation": "STK_builtin_AI_explicit_PlayerConfig",
            "on_screen_names": settings.show_names,
        }

    def assets(self):
        p = self.p
        return {"tracks": list(p.list_tracks(p.RaceConfig.RaceMode.NORMAL_RACE)),
                "karts": list(p.list_karts()), "provenance": self.provenance}

    def stop_race(self):
        if self.race is not None:
            self.world = None
            self.race.stop()
            self.race = None
            gc.collect()

    def reset(self, settings: RaceSettings, participants: list[Participant]):
        validate_participants(settings, participants)
        old = self.settings
        if (old.width, old.height, old.quality) != (settings.width, settings.height, settings.quality):
            raise ValueError("Resolución/calidad inmutables por proceso; crea otro Engine")
        self.stop_race()
        self.settings, self.participants = settings, sorted(participants, key=lambda x: x.slot)
        self.provenance = {**self.provenance, "on_screen_names": settings.show_names}
        p = self.p
        assets = self.assets()
        if settings.track not in assets["tracks"]:
            raise ValueError(f"Pista desconocida: {settings.track}; ejecuta assets")
        by_slot = {x.slot: x for x in self.participants}
        config = p.RaceConfig()
        for key, value in {"track": settings.track, "laps": settings.laps,
                           "seed": settings.resolved_seed, "reverse": settings.reverse,
                           "difficulty": settings.difficulty, "num_kart": settings.num_karts,
                           "step_size": 1.0/settings.tick_hz, "overlay": settings.overlay,
                           "num_cameras": 0}.items():
            if not hasattr(config, key):
                raise RuntimeError(f"Falta RaceConfig.{key}; binding incompatible")
            setattr(config, key, value)
        from .roster import select_roster
        self.roster = select_roster(settings, self.participants, assets["karts"])
        # IMPORTANT pinned-binding compatibility:
        # Native RaceManager puts implicit AI first and players last, while
        # PySTKRace::start uses getKart(i) for configured players' cameras and
        # AI controllers. Mixing implicit bots here can point a camera at the
        # wrong kart. Explicit slots preserve the tested index contract.
        # These controllers use the built-in game AI.
        # Empty display names remove artificial bot_X labels without HUD edits.
        # num_cameras=0: >0 follows leaders instead of fixed agents.
        while len(config.players) < settings.num_karts:
            config.players.append(p.PlayerConfig())
        for slot in range(settings.num_karts):
            participant = by_slot.get(slot)
            pc = config.players[slot]
            pc.kart = self.roster[slot]["kart"]
            pc.name = (participant.name if participant else pc.kart) if settings.show_names else ""
            external = participant is not None and participant.kind == "external"
            pc.controller = (p.PlayerConfig.Controller.PLAYER_CONTROL if external
                             else p.PlayerConfig.Controller.AI_CONTROL)
            camera = participant is not None and participant.observe
            pc.camera_mode = p.PlayerConfig.CameraMode.ON if camera else p.PlayerConfig.CameraMode.OFF
        self.observed = [x for x in self.participants if x.observe]
        self.controlled = [x for x in self.participants if x.kind == "external"]
        self.race = p.Race(config)
        self.race.start()
        self.world = p.WorldState()
        # Bootstrap documentado: genera los primeros buffers antes de la observación 0.
        self.running = bool(self.race.step([p.Action() for _ in self.controlled])
                            if self.controlled else self.race.step())
        self.frame_id = 0
        track = p.Track()
        track.update()  # Solo una vez; no acumulamos successors con updates repetidos.
        successors = list(track.successors)
        offsets = np.concatenate(([0], np.cumsum([len(x) for x in successors]))).astype(np.int64)
        self.track = {
            "length": np.asarray(track.length, np.float32),
            "path_nodes": np.asarray(track.path_nodes, np.float32).copy(),
            "path_width": np.asarray(track.path_width, np.float32).reshape(-1).copy(),
            "path_distance": np.asarray(track.path_distance, np.float32).copy(),
            "successor_offsets": offsets,
            "successor_values": np.asarray([v for x in successors for v in x], np.int32),
        }
        snap = self.snapshot()
        # Validate state-slot mapping rather than assuming a successful render
        # proves that every camera/controller observes the intended kart.
        for participant in self.participants:
            k = next((x for x in snap["state"]["karts"] if x["id"] == participant.slot), None)
            if k is None:
                raise RuntimeError(f"No existe slot {participant.slot} en WorldState")
            expected_kart = self.roster[participant.slot]["kart"]
            if k.get("kart_type") is not None and k["kart_type"] != expected_kart:
                raise RuntimeError(f"Kart del slot {participant.slot}: {k['kart_type']}, esperado {expected_kart}")
        return {"snapshot": snap, "track": self.track, "provenance": self.provenance,
                "participants": [asdict(x) for x in self.participants], "roster": self.roster}

    def snapshot(self, render=True):
        self.world.update()
        rgb = {}
        if render:
            data = self.race.render_data
            if len(data) != len(self.observed):
                raise RuntimeError(f"Cámaras: {len(data)}, esperadas: {len(self.observed)}. No se reasignan silenciosamente.")
            for camera, participant in zip(data, self.observed):
                image = np.asarray(camera.image).copy(order="C")
                check_rgb(image, self.settings.width, self.settings.height)
                rgb[participant.name] = image
        return {"frame_id": self.frame_id, "sim_time": self.frame_id/self.settings.tick_hz,
                "rgb": rgb, "state": extract_world(self.world), "running": self.running}

    def step(self, actions: dict, render=True):
        if self.race is None:
            raise RuntimeError("Ejecuta reset() antes de step()")
        expected = {x.name for x in self.controlled}
        if set(actions) != expected:
            raise ValueError(f"Acciones esperadas para {expected}, recibidas para {set(actions)}")
        native = [self.p.Action(**Action.from_any(actions[x.name]).as_dict()) for x in self.controlled]
        self.running = bool(self.race.step(native) if native else self.race.step())
        self.frame_id += 1
        return self.snapshot(render)

    def close(self):
        self.stop_race()
        self.p.clean()

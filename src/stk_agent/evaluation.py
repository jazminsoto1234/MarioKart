from __future__ import annotations
from pathlib import Path
import itertools
import json
import time
import numpy as np
from .agents import AgentWorker
from .config import RaceSettings, Participant
from .engine import Engine
from .env import kart_state
from .types import Action
from .util import atomic_json
from .video import VideoWriter


def mosaic(images):
    if len(images) == 1:
        return images[0]
    blank = np.zeros_like(images[0])
    images = list(images) + [blank]*(4-len(images))
    return np.concatenate((np.concatenate(images[:2], axis=1),
                           np.concatenate(images[2:4], axis=1)), axis=0)


def run_race(manifests, settings, output, slots=None, show=False,
             realtime=False, video=False, action_timeout=10.0, engine_factory=Engine):
    """realtime solo limita el ritmo máximo; NO descarta acciones que llegan tarde.

    Un timeout invalida la carrera, no se sustituye con un bot o una heurística.
    Para una evaluación temporal estricta debe definirse una política de tiempo explícita.
    """
    if not 1 <= len(manifests) <= 4:
        raise ValueError("Se admiten de uno a cuatro modelos por carrera")
    if action_timeout <= 0:
        raise ValueError("action_timeout debe ser positivo")
    slots = list(range(len(manifests))) if slots is None else list(slots)
    if len(slots) != len(manifests) or len(set(slots)) != len(slots):
        raise ValueError("Un slot diferente por modelo")
    if any(type(s) is not int or s < 0 or s >= settings.num_karts for s in slots):
        raise ValueError("Slot inválido")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    workers, writers, ui = [], [], None
    engine = None
    report = {"status": "initializing", "settings": settings.as_dict(),
              "manifests": [str(Path(m).resolve()) for m in manifests], "slots": slots,
              "mode": "lock_step_paced" if realtime else "lock_step"}
    atomic_json(output/"result.json", report)
    try:
        for i, manifest in enumerate(manifests):
            workers.append(AgentWorker(manifest, output/f"agent_{i}.log"))
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError("Cada modelo debe tener un name único en su manifest")
        participants = [Participant(name, slot) for name, slot in zip(names, slots)]
        engine = engine_factory(settings)
        start = engine.reset(participants)
        snap = start["snapshot"]
        report["engine"] = start["provenance"]
        report["roster"] = start.get("roster", [])
        for worker in workers:
            worker.reset()
        if show:
            from .input import HumanUI
            multiplier = 1 if len(workers) == 1 else 2
            ui = HumanUI(settings.width*multiplier, settings.height*multiplier, scale=1.0)
        if video:
            for i in range(len(workers)):
                writers.append(VideoWriter(output/f"view_{i}.mp4", settings.width, settings.height, settings.tick_hz))
        current_actions = {name: Action().as_dict() for name in names}
        finished = {name: False for name in names}
        finish_states = {}
        started = time.perf_counter()
        status = "time_limit"
        while True:
            for name, slot in zip(names, slots):
                if not finished[name] and kart_state(snap["state"], slot)["has_finished_race"]:
                    finished[name] = True
                    finish_states[name] = dict(kart_state(snap["state"], slot))
            images = [snap["rgb"][name] for name in names]
            if ui:
                ui.show(mosaic(images), "Competencia | cámaras independientes | ESC termina exhibición")
                ui.events()
                if ui.quit:
                    status = "user_abort"
                    break
            for writer, image in zip(writers, images):
                writer.write(image)
            if all(finished.values()):
                status = "finished"
                break
            if snap["sim_time"] >= settings.max_seconds or not snap["running"]:
                break
            if snap["frame_id"] % settings.action_repeat == 0:
                # Enviar a todos ANTES de esperar: mismo instante, sin información mutua.
                deadline = time.perf_counter()+action_timeout
                for worker in workers:
                    if not finished[worker.name]:
                        worker.request(snap["rgb"][worker.name])
                for worker in workers:
                    if not finished[worker.name]:
                        current_actions[worker.name] = worker.receive(deadline).as_dict()
                    else:
                        current_actions[worker.name] = Action().as_dict()
            if realtime:
                delay = started+(snap["frame_id"]+1)/settings.tick_hz-time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            snap = engine.step(current_actions, render=True)
        elapsed = time.perf_counter()-started
        results = []
        for name, slot, worker in zip(names, slots, workers):
            k = finish_states.get(name, kart_state(snap["state"], slot))
            results.append({"name": name, "slot": slot, "finished": bool(k["has_finished_race"]),
                            "native_position": k["position"], "finish_time": k["finish_time"] if k["has_finished_race"] else None,
                            "overall_distance": k["overall_distance"], "timing": worker.timing()})
        report.update(status=status, results=results, simulated_seconds=snap["sim_time"],
                      wall_seconds=elapsed, realtime_factor=snap["sim_time"]/elapsed if elapsed else None)
        atomic_json(output/"result.json", report)
    except BaseException as e:
        report.update(status="invalid", error=f"{type(e).__name__}: {e}")
        atomic_json(output/"result.json", report)
        raise
    finally:
        close_errors = []
        for writer in writers:
            try:
                writer.close()
            except Exception as e:
                close_errors.append(str(e))
        for worker in workers:
            worker.close()
        if engine:
            engine.close()
        if ui:
            ui.close()
        if close_errors:
            report["video_errors"] = close_errors
            atomic_json(output/"result.json", report)
    return report


def league_schedule(manifests, tracks, seeds, heat_size=4):
    """Todos los subconjuntos de heat_size; rota todos los slots controlados.

    Se exportan resultados brutos de todas las mangas.
    El número de carreras crece combinatoriamente: revisar schedule antes.
    """
    if not 1 <= heat_size <= 4 or len(manifests) < heat_size:
        raise ValueError("Tamaño de manga inválido")
    schedule = []
    for combination in itertools.combinations(manifests, heat_size):
        for track in tracks:
            for seed in seeds:
                for rotation in range(heat_size):
                    schedule.append({"manifests": list(combination), "track": track, "seed": seed,
                                     "slots": [(i+rotation) % heat_size for i in range(heat_size)]})
    return schedule


def run_league(manifests, settings, output, tracks, seeds, heat_size=4, execute=False, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    schedule = league_schedule(manifests, tracks, seeds, heat_size)
    atomic_json(output/"schedule.json", schedule)
    if not execute:
        return {"races": len(schedule), "executed": False, "schedule": str(output/"schedule.json")}
    results = []
    for index, item in enumerate(schedule):
        result = run_race(item["manifests"], settings.updated(track=item["track"], seed=item["seed"]),
                          output/f"race_{index:05d}", slots=item["slots"], **kwargs)
        results.append(result)
        atomic_json(output/"results.json", results)
    return {"races": len(schedule), "executed": True}

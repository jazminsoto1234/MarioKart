from __future__ import annotations
from pathlib import Path
import importlib.metadata
import platform
import time
import numpy as np
from .config import Participant
from .engine import Engine
from .util import atomic_json
from .video import ffmpeg_executable


def package_report():
    packages = {}
    for distribution in ('stk-agent', 'PySuperTuxKart2', 'numpy', 'h5py', 'pygame', 'gymnasium', 'imageio-ffmpeg'):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        ffmpeg = ffmpeg_executable()
    except Exception as e:
        ffmpeg = str(e)
    from . import __version__
    return {
        'library_source_version': __version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': packages,
        'ffmpeg': ffmpeg,
        'note': 'Instalación y versiones no equivalen a una prueba de renderizado',
    }


def write_ppm(path, image):
    image = np.asarray(image, np.uint8)
    h, w, _ = image.shape
    with Path(path).open('wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode())
        f.write(image.tobytes())


def probe(settings, output, seconds=8, views=1, show=False):
    """Live integration diagnostic for RGB/cameras only."""
    if not 1 <= views <= min(4, settings.num_karts) or seconds <= 0:
        raise ValueError('views/seconds inválidos')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = package_report()
    ui = None
    result['requested_settings'] = settings.as_dict()
    engine_settings = settings.updated(seed=42) if settings.seed == 'random' else settings
    result['engine_settings'] = engine_settings.as_dict()
    try:
        with Engine(engine_settings) as engine:
            players = [Participant(f'probe_{i}', i, kind='bot') for i in range(views)]
            start = engine.reset(players)
            snap = start['snapshot']
            result['engine'] = start['provenance']
            result['roster'] = start.get('roster', [])
            checks = {p.name: {'rgb_changes': False} for p in players}
            initial = {name: image.copy() for name, image in snap['rgb'].items()}
            if show:
                from .input import HumanUI
                multiple = 1 if views == 1 else 2
                ui = HumanUI(settings.width * multiple, settings.height * multiple, scale=1)
            started = time.perf_counter()
            steps = max(1, int(seconds * settings.tick_hz))
            for _ in range(steps):
                snap = engine.step({}, render=True)
                for name, image in snap['rgb'].items():
                    checks[name]['rgb_changes'] |= bool(np.any(image != initial[name]))
                if ui:
                    from .evaluation import mosaic
                    ui.show(mosaic([snap['rgb'][p.name] for p in players]),
                            'Diagnóstico: comprobar orientación, HUD y cámaras')
                    ui.events()
                    if ui.quit:
                        break
            elapsed = time.perf_counter() - started
            valid = all(c['rgb_changes'] for c in checks.values())
            result.update(
                checks=checks,
                native_steps=snap['frame_id'],
                wall_seconds=elapsed,
                capture_steps_per_second=snap['frame_id'] / elapsed,
                realtime_factor=snap['sim_time'] / elapsed,
                valid=valid,
                valid_scope='RGB stepping + camera assignment',
                measured_workload='RGB+privileged numerical state each step; WITHOUT recorder/video encoder',
                hud_and_camera_assignment='REQUIRES_VISUAL_CONFIRMATION',
            )
            for name, image in snap['rgb'].items():
                write_ppm(output / f'{name}.ppm', image)
    except Exception as e:
        result.update(valid=False, error=f'{type(e).__name__}: {e}')
    finally:
        if ui:
            ui.close()
    atomic_json(output / 'doctor.json', result)
    return result

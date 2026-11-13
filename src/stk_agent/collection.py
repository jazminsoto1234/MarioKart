from __future__ import annotations
import collections
import random
import secrets
import shutil
import time
from pathlib import Path
from .config import Participant, RANDOM
from .engine import Engine
from .env import kart_state
from .input import HumanUI
from .recording import Recorder
from .transfer import publish


def resolve_collection_episode(settings, episode_number, track=None):
    """Resuelve seed y slot sin ocultarlos: el cfg devuelto es totalmente reproducible."""
    if type(episode_number) is not int or episode_number < 0:
        raise ValueError('episode_number inválido')
    if settings.seed == RANDOM:
        seed = secrets.randbelow(2**31)
    else:
        seed = (int(settings.seed) + episode_number) % (2**31)
    if settings.player_slot == RANDOM:
        # Independiente de la selección de karts, pero reproducible a partir del seed efectivo.
        slot = random.Random(seed ^ 0x53544B).randrange(settings.num_karts)
    else:
        slot = int(settings.player_slot)
    return settings.updated(track=track or settings.track, seed=seed, player_slot=slot), slot


def collect(settings, root, group_id, driver_id, episodes=1, tracks=None,
            joystick=None, mapping=None, keymap=None, scale=1.5, codec='h264', publish_dir=None):
    if episodes < 1 or not group_id or not driver_id:
        raise ValueError('episodes, group_id y driver_id son obligatorios/válidos')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tracks = tracks or [settings.track]
    saved = []
    with Engine(settings) as engine:
        # Pygame belongs to the human process; OpenGL/STK belongs to its child.
        ui = HumanUI(settings.width, settings.height, scale, joystick, keymap=keymap, mapping=mapping)
        try:
            for episode_number in range(episodes):
                if shutil.disk_usage(root).free < 512 * 1024**2:
                    raise RuntimeError('Menos de 512 MiB libres. Libera espacio antes de grabar')
                requested_track = tracks[episode_number % len(tracks)]
                cfg, human_slot = resolve_collection_episode(settings, episode_number, requested_track)
                participant = Participant('human', human_slot)
                start = engine.reset([participant], cfg)
                snap = start['snapshot']
                if start.get('roster'):
                    print('Personajes: ' + ', '.join(f"{r['slot']}:{r['kart']}" for r in start['roster']), flush=True)
                print(
                    f'Grabación: seed={cfg.seed}, slot_humano={human_slot}, pantalla={cfg.tick_hz} Hz, '
                    f'RGB={cfg.rgb_hz} Hz, calidad={cfg.quality}', flush=True
                )
                metadata = {
                    'group_id': group_id,
                    'driver_id': driver_id,
                    'source': 'human',
                    'engine': start['provenance'],
                    'observation_contract': 'RGB_WITH_HUD_v1',
                    'training_data_contract': 'RGB_ACTION_PRIVILEGED_STATE_v1',
                    'input_device': 'gamepad' if joystick is not None else 'keyboard',
                    'input_keymap': ui.keymap,
                    'gamepad_mapping': ui.mapping if joystick is not None else None,
                    'keyboard_steer_rate_per_second': ui.keyboard_steer_rate,
                    'audio_in_policy': False,
                    'human_slot': human_slot,
                    'effective_seed': cfg.seed,
                    'collection_randomization': {
                        'requested_seed': settings.seed,
                        'effective_seed': cfg.seed,
                        'requested_player_slot': settings.player_slot,
                        'effective_player_slot': human_slot,
                    },
                    'roster': start.get('roster', []),
                    'participants': start.get('participants', []),
                }
                recorder = Recorder(root, metadata, cfg, start['track'], 'human', codec=codec)
                started = time.perf_counter()
                period = 1 / cfg.tick_hz
                deadline = started + period
                paused = 0.0
                overruns = 0
                phases = collections.Counter()
                status = 'user_abort'
                try:
                    while not ui.quit:
                        display_ns = ui.show(
                            snap['rgb']['human'],
                            f'{cfg.track} | {episode_number+1}/{episodes} | {group_id}/{driver_id} | ESC termina',
                        )
                        ui.wait_until(deadline)
                        pause = ui.pause_if_unfocused()
                        if pause:
                            paused += pause
                            deadline = time.perf_counter() + period
                            ui.wait_until(deadline)
                        if ui.quit:
                            break
                        action = ui.action()
                        input_ns = time.perf_counter_ns()
                        applied_ns = time.perf_counter_ns()
                        after = engine.step({'human': action.as_dict()}, render=True)
                        ego = kart_state(after['state'], human_slot)
                        terminated = bool(ego['has_finished_race'])
                        truncated = not terminated and (after['sim_time'] >= cfg.max_seconds or not after['running'])
                        recorder.append(
                            snap, action,
                            terminated=terminated,
                            truncated=truncated,
                            display_time_ns=display_ns,
                            input_time_ns=input_ns,
                            applied_time_ns=applied_ns,
                        )
                        phases[str(snap['state'].get('phase'))] += 1
                        snap = after
                        deadline += period
                        now = time.perf_counter()
                        if now > deadline + period:
                            overruns += 1
                            deadline = now + period  # Never drop simulation ticks to catch up.
                        if terminated or truncated:
                            status = 'finished' if terminated else 'time_limit'
                            break
                        if snap['frame_id'] % cfg.tick_hz == 0 and shutil.disk_usage(root).free < 256 * 1024**2:
                            status = 'low_disk_space'
                            break
                    if snap['frame_id'] < 1:
                        recorder.abort('El usuario cerró antes de la primera transición')
                        break
                    wall = time.perf_counter() - started
                    ego = kart_state(snap['state'], human_slot)
                    result = {
                        'status': status,
                        'finished': bool(ego['has_finished_race']),
                        'native_position': ego['position'],
                        'finish_time': ego['finish_time'],
                        'overall_distance': ego['overall_distance'],
                    }
                    active_wall = max(wall - paused, 1e-9)
                    quality = {
                        'wall_seconds': wall,
                        'paused_wall_seconds': paused,
                        'active_wall_seconds': active_wall,
                        'effective_steps_per_second': snap['frame_id'] / active_wall,
                        'realtime_factor': snap['sim_time'] / active_wall,
                        'pacer_overruns': overruns,
                        'phase_frame_counts': dict(phases),
                        'dropped_frames': 0,
                        'hud_and_camera_alignment': 'must_be_checked_with_doctor_and_inspect',
                    }
                    path = recorder.finish(snap, result, quality)
                    saved.append(str(path))
                    print(f'Episodio validado: {path}', flush=True)
                    if publish_dir:
                        destination = publish(path, publish_dir)
                        print(f'Copiado a carpeta sincronizada: {destination}. Verifica el estado de Google Drive.', flush=True)
                    if ui.quit or status == 'low_disk_space':
                        break
                except BaseException as e:
                    recorder.abort(str(e))
                    raise
        finally:
            ui.close()
    return saved

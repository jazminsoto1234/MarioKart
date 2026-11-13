from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import os
import queue
import threading
import uuid
import h5py
import numpy as np
from .types import ACTION_NAMES, Action, check_rgb
from .video import VideoWriter, iter_video
from .util import atomic_json, jsonable, sha256, safe_relative

SCHEMA_VERSION = '3.0.0'
COMPATIBLE_SCHEMA_VERSIONS = {'2.0.0', SCHEMA_VERSION}


def _append(dataset, value):
    n = len(dataset)
    dataset.resize(n + 1, axis=0)
    dataset[n] = value


def _extend(dataset, values):
    n = len(dataset)
    dataset.resize(n + len(values), axis=0)
    dataset[n:] = values


class EpisodeWriter:
    """Single writer, owned by the recorder thread (not shared across threads)."""
    def __init__(self, root, metadata, settings, track, participant, codec='h264'):
        from . import __version__
        self.metadata = copy.deepcopy(metadata)
        self.metadata.update(
            schema_version=SCHEMA_VERSION,
            library_version=__version__,
            episode_id=uuid.uuid4().hex,
            created_utc=datetime.now(timezone.utc).isoformat(),
            settings=settings.as_dict(),
            participant=participant,
            action_names=ACTION_NAMES,
            codec=codec,
            rgb_lossless=codec == 'ffv1',
            data_contract='RGB_ACTION_PRIVILEGED_STATE_v1',
            sampling={
                'api_tick_hz': settings.tick_hz,
                'human_display_target_hz': settings.tick_hz,
                'actions_hz': settings.tick_hz,
                'state_hz': settings.tick_hz,
                'rgb_nominal_hz': settings.rgb_hz,
                'policy_hz': settings.policy_hz,
                'physics_hz_reference': 120,
                'time_authority': 'HDF5 frames/sim_time indexed by rgb/frame_id; not MP4 timestamps',
                'final_rgb': 'terminal state always saved; final interval may be shorter than 1/rgb_hz',
            },
        )
        self.settings, self.participant = settings, participant
        self.final_path = Path(root) / self.metadata['episode_id']
        self.path = self.final_path.with_name(self.final_path.name + '.partial')
        self.path.mkdir(parents=True, exist_ok=False)
        self.frames = self.transitions = self.rgb_frames = 0
        self.closed = False
        self.width, self.height = settings.width, settings.height
        atomic_json(self.path / 'meta.json', self.metadata)
        np.savez_compressed(self.path / 'track.npz', **track)
        self.h5 = h5py.File(self.path / 'data.h5', 'w')
        self.h5.attrs['schema_version'] = SCHEMA_VERSION
        for key, dtype in [
            ('frames/id', 'i8'), ('frames/sim_time', 'f8'), ('frames/display_time_ns', 'i8'),
            ('frames/phase', 'S40'), ('rgb/frame_id', 'i8'),
            ('actions/frame_id', 'i8'), ('actions/input_time_ns', 'i8'),
            ('actions/applied_time_ns', 'i8'), ('actions/terminated', '?'), ('actions/truncated', '?'),
        ]:
            self.h5.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1024,))
        self.h5.create_dataset('actions/value', shape=(0, 7), maxshape=(None, 7),
                               dtype='f4', chunks=(1024, 7))
        self.h5.create_dataset('privileged/state_bytes', shape=(0,), maxshape=(None,), dtype='u1',
                               chunks=(262144,), compression='gzip', compression_opts=1)
        offsets = self.h5.create_dataset('privileged/state_offsets', shape=(1,), maxshape=(None,),
                                         dtype='i8', chunks=(1024,))
        offsets[0] = 0
        self.metadata['rgb_file'] = 'rgb.mp4' if codec == 'h264' else 'rgb.mkv'
        try:
            self.video = VideoWriter(self.path / self.metadata['rgb_file'], self.width, self.height,
                                     settings.rgb_hz, codec)
        except BaseException:
            self.h5.close()
            raise

    def _frame(self, snap, display_time_ns=0, terminal=False):
        tick = snap['frame_id']
        if tick != self.frames:
            raise ValueError(f'Tick {tick}, esperado {self.frames}. No se permiten saltos')
        if self.settings.wants_rgb(tick) or terminal:
            rgb = check_rgb(snap['rgb'][self.participant], self.width, self.height)
            self.video.write(rgb)
            _append(self.h5['rgb/frame_id'], tick)
            self.rgb_frames += 1
        _append(self.h5['frames/id'], tick)
        _append(self.h5['frames/sim_time'], snap['sim_time'])
        _append(self.h5['frames/display_time_ns'], display_time_ns)
        _append(self.h5['frames/phase'], str(snap['state'].get('phase', 'UNKNOWN')).encode('ascii'))
        data = json.dumps(jsonable(snap['state']), separators=(',', ':'), allow_nan=False).encode('utf-8')
        target = self.h5['privileged/state_bytes']
        _extend(target, np.frombuffer(data, dtype=np.uint8))
        _append(self.h5['privileged/state_offsets'], len(target))
        self.frames += 1
        if self.frames % self.settings.tick_hz == 0:
            self.h5.attrs['committed_state_frames'] = self.frames
            self.h5.flush()

    def append(self, snap, action, terminated=False, truncated=False,
               display_time_ns=0, input_time_ns=0, applied_time_ns=0):
        action = Action.from_any(action)
        if self.frames != self.transitions:
            raise RuntimeError('No se puede agregar después del estado terminal')
        if terminated and truncated:
            raise ValueError('terminated y truncated no pueden ser ambos true')
        self._frame(snap, display_time_ns)
        for key, value in [
            ('frame_id', snap['frame_id']), ('value', action.as_array()),
            ('input_time_ns', input_time_ns), ('applied_time_ns', applied_time_ns),
            ('terminated', terminated), ('truncated', truncated),
        ]:
            _append(self.h5['actions/' + key], value)
        self.transitions += 1

    def finish(self, final_snapshot, result, quality=None):
        if self.closed:
            raise RuntimeError('Episodio ya finalizado')
        if self.transitions and not self.h5['actions/terminated'][-1] and not self.h5['actions/truncated'][-1]:
            finished = bool(result.get('finished', result.get('status') == 'finished'))
            self.h5['actions/terminated'][-1] = finished
            self.h5['actions/truncated'][-1] = not finished
        self._frame(final_snapshot, terminal=True)
        self.h5.attrs['committed_state_frames'] = self.frames
        self.h5.close()
        self.video.close()
        self.closed = True
        self.metadata.update(
            num_frames=self.frames,
            num_transitions=self.transitions,
            num_state_frames=self.frames,
            num_rgb_frames=self.rgb_frames,
            num_visual_transitions=self.rgb_frames - 1,
            result=jsonable(result),
            quality=quality or {},
            simulated_seconds=self.transitions / self.settings.tick_hz,
        )
        atomic_json(self.path / 'meta.json', self.metadata)
        files = ['meta.json', 'data.h5', 'track.npz', self.metadata['rgb_file']]
        atomic_json(self.path / 'checksums.json', {name: sha256(self.path / name) for name in files})
        report = validate_episode(self.path, verify_hashes=True, decode_video=True, allow_partial=True)
        if not report['valid']:
            atomic_json(self.path / 'validation_error.json', report)
            raise RuntimeError(f"Episodio inválido: {report['errors']}")
        atomic_json(self.path / 'READY.json', {
            'schema_version': SCHEMA_VERSION,
            'checksums_sha256': sha256(self.path / 'checksums.json'),
        })
        os.replace(self.path, self.final_path)
        return self.final_path

    def abort(self, message):
        if not self.closed:
            try:
                self.h5.close()
            finally:
                try:
                    self.video.close()
                except Exception:
                    pass
            self.closed = True
        atomic_json(self.path / 'ERROR.json', {'error': message, 'ticks_written': self.frames})


class Recorder:
    """Cola acotada; append copia los buffers para que no cambien al avanzar STK."""
    def __init__(self, root, metadata, settings, track, participant, codec='h264', queue_size=12):
        self.commands = queue.Queue(maxsize=queue_size)
        self.error = None
        self.result = None
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            args=(root, metadata, settings, track, participant, codec),
            daemon=True,
            name='episode-writer',
        )
        self.thread.start()
        if not self.ready.wait(30):
            raise TimeoutError('No se pudo iniciar el recorder')
        self._check()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('El recorder falló: ' + str(self.error)) from self.error

    def _run(self, *args):
        writer = None
        try:
            writer = EpisodeWriter(*args)
            self.ready.set()
            while True:
                kind, payload = self.commands.get()
                if kind == 'append':
                    writer.append(**payload)
                elif kind == 'finish':
                    self.result = writer.finish(**payload)
                    break
                elif kind == 'abort':
                    writer.abort(payload['message'])
                    break
        except BaseException as e:
            self.error = e
            if writer is not None:
                try:
                    writer.abort(str(e))
                except Exception:
                    pass
        finally:
            self.ready.set()

    def append(self, snap, action, **kwargs):
        self._check()
        payload = {'snap': copy.deepcopy(snap), 'action': Action.from_any(action), **kwargs}
        try:
            self.commands.put_nowait(('append', payload))
        except queue.Full as e:
            raise RuntimeError('Grabación demasiado lenta: cola llena. Episodio detenido; no se eliminan frames.') from e

    def _end(self, command, payload):
        while self.thread.is_alive():
            self._check()
            try:
                self.commands.put((command, payload), timeout=0.2)
                break
            except queue.Full:
                continue
        self.thread.join(180)
        if self.thread.is_alive():
            raise TimeoutError('El writer no terminó. No subas la carpeta .partial')
        self._check()
        return self.result

    def finish(self, final_snapshot, result, quality=None):
        return self._end('finish', {'final_snapshot': copy.deepcopy(final_snapshot),
                                    'result': result, 'quality': quality})

    def abort(self, message):
        if self.thread.is_alive() and self.error is None:
            self._end('abort', {'message': str(message)})


def _core_settings(meta):
    settings = meta.get('settings', {})
    tick_hz = int(settings['tick_hz'])
    rgb_hz = int(settings.get('rgb_hz', tick_hz))
    width, height = int(settings['width']), int(settings['height'])
    if tick_hz <= 0 or rgb_hz <= 0 or tick_hz % rgb_hz:
        raise ValueError('Frecuencias almacenadas inválidas')
    return tick_hz, rgb_hz, width, height


def validate_episode(path, verify_hashes=True, decode_video=False, allow_partial=False):
    """Validate current schema and read-only compatibility with schema 1/2.

    Schema 2 may contain depth/instance groups; they are ignored rather than
    exposing them through the current API.
    """
    path = Path(path)
    errors, warnings = [], []
    report = {'path': str(path), 'valid': False, 'errors': errors, 'warnings': warnings}
    try:
        meta = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
        schema = meta['schema_version']
        if schema == '1.0.0':
            from ._legacy_validation import validate_episode as legacy
            return legacy(path, verify_hashes, decode_video, allow_partial)
        if schema not in COMPATIBLE_SCHEMA_VERSIONS:
            raise ValueError(f'Versión de esquema no soportada: {schema}')
        if not allow_partial and (path.name.endswith('.partial') or not (path / 'READY.json').is_file()):
            raise ValueError('Falta READY.json o el episodio sigue siendo .partial')
        n, t, nrgb = meta['num_state_frames'], meta['num_transitions'], meta['num_rgb_frames']
        if n != t + 1 or t < 1 or nrgb < 2 or meta['num_frames'] != n:
            raise ValueError('Se requieren T>=1 controles, T+1 estados y al menos dos RGB')
        if meta['num_visual_transitions'] != nrgb - 1:
            raise ValueError('Cantidad visual inconsistente')
        tick_hz, rgb_hz, width, height = _core_settings(meta)
        if schema == SCHEMA_VERSION:
            removed = {'depth_hz','record_depth','depth_storage','instance_hz','record_instance','auxiliary_hz'}
            bad = removed.intersection(meta.get('settings', {}))
            if bad:
                raise ValueError('Schema 3 no admite canales geométricos: ' + ', '.join(sorted(bad)))
        if verify_hashes:
            hashes = json.loads((path / 'checksums.json').read_text(encoding='utf-8'))
            required = {'meta.json', 'data.h5', 'track.npz', meta['rgb_file']}
            if not required.issubset(hashes):
                raise ValueError('Faltan archivos en checksums.json')
            for name, expected in hashes.items():
                if sha256(safe_relative(path, name)) != expected:
                    raise ValueError(f'SHA256 incorrecto: {name}')
            if (path / 'READY.json').exists():
                ready = json.loads((path / 'READY.json').read_text())
                if ready['checksums_sha256'] != sha256(path / 'checksums.json'):
                    raise ValueError('READY/checksums no coinciden')
        with h5py.File(path / 'data.h5', 'r') as f:
            if str(f.attrs.get('schema_version')) != schema:
                raise ValueError('HDF5/meta schema diferente')
            if schema == SCHEMA_VERSION and ('depth' in f or 'instance' in f or 'aux' in f):
                raise ValueError('Schema 3 no debe contener depth/instance')
            if schema == '2.0.0' and ('depth' in f or 'instance' in f):
                warnings.append('Episodio schema 2: depth/instance históricos se ignoran')
            if not np.array_equal(f['frames/id'][:], np.arange(n)) or not np.array_equal(f['actions/frame_id'][:], np.arange(t)):
                raise ValueError('IDs de tick/acción no contiguos')
            for key in ('value','terminated','truncated','input_time_ns','applied_time_ns'):
                if len(f['actions/' + key]) != t:
                    raise ValueError(f'Cantidad actions/{key} incorrecta')
            for key in ('sim_time','phase','display_time_ns'):
                if len(f['frames/' + key]) != n:
                    raise ValueError(f'Cantidad frames/{key} incorrecta')
            if not np.allclose(f['frames/sim_time'][:], np.arange(n) / tick_hz, rtol=0, atol=1e-8):
                raise ValueError('Timestamps simulados incompatibles')
            actions = f['actions/value'][:]
            if actions.shape != (t, 7) or not np.isfinite(actions).all():
                raise ValueError('Acciones con shape incorrecto o NaN/Inf')
            if (np.any(np.abs(actions[:,0]) > 1) or np.any((actions[:,1] < 0) | (actions[:,1] > 1))
                    or not np.isin(actions[:,2:], [0,1]).all()):
                raise ValueError('Acciones fuera de rango')
            term, trunc = f['actions/terminated'][:], f['actions/truncated'][:]
            if term[:-1].any() or trunc[:-1].any() or (term & trunc).any() or not (term[-1] or trunc[-1]):
                raise ValueError('Terminación/truncamiento incorrecto')
            offsets = f['privileged/state_offsets'][:]
            if (len(offsets) != n + 1 or offsets[0] != 0 or (np.diff(offsets) <= 0).any()
                    or offsets[-1] != len(f['privileged/state_bytes'])):
                raise ValueError('Offsets/estado inválidos')
            rgb_ids = f['rgb/frame_id'][:]
            expected = np.unique(np.r_[np.arange(0, n, tick_hz // rgb_hz), t])
            if len(rgb_ids) != nrgb or not np.array_equal(rgb_ids, expected):
                raise ValueError('RGB/tick no coincide con muestreo + observación terminal')
        if decode_video:
            decoded = sum(1 for _ in iter_video(safe_relative(path, meta['rgb_file']), width, height))
            if decoded != nrgb:
                raise ValueError(f'Video: {decoded}, esperado {nrgb}')
        report.update(valid=True, frames=n, rgb_frames=nrgb, transitions=t,
                      visual_transitions=nrgb-1, source=meta.get('source'),
                      episode_id=meta['episode_id'], simulated_seconds=meta['simulated_seconds'],
                      schema_version=schema)
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}')
    return report

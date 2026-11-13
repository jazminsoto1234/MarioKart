from __future__ import annotations
from collections import deque
from pathlib import Path
import json
import h5py
import numpy as np
from .recording import validate_episode
from .video import iter_video
from .util import safe_relative

DRIVING_PHASES = ('GO_PHASE', 'MUSIC_PHASE', 'RACE_PHASE')
CONTROL_PHASES = ('READY_PHASE', 'SET_PHASE') + DRIVING_PHASES


def _phase_filter(phases):
    if phases is None:
        return None
    if isinstance(phases, str):
        raise ValueError('phases debe ser una colección, no un string')
    return set(phases)


class Episode:
    def __init__(self, path, validate=True):
        self.path = Path(path)
        if validate:
            report = validate_episode(self.path, verify_hashes=False)
            if not report['valid']:
                raise ValueError(report['errors'])
        self.meta = json.loads((self.path / 'meta.json').read_text(encoding='utf-8'))
        self.settings = self.meta['settings']
        self.num_ticks = int(self.meta['num_transitions'])
        self.legacy = self.meta['schema_version'] == '1.0.0'
        with h5py.File(self.path / 'data.h5', 'r') as f:
            self.rgb_tick_ids = np.arange(self.num_ticks + 1) if self.legacy else f['rgb/frame_id'][:]
        self.length = len(self.rgb_tick_ids) - 1

    def __len__(self):
        return self.length

    def load_track(self):
        with np.load(self.path / 'track.npz', allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}

    @staticmethod
    def _state(f, index):
        a, b = f['privileged/state_offsets'][index:index+2]
        return json.loads(f['privileged/state_bytes'][a:b].tobytes())

    def state_at_tick(self, tick):
        if not 0 <= tick <= self.num_ticks:
            raise IndexError(tick)
        with h5py.File(self.path / 'data.h5', 'r') as f:
            return self._state(f, tick)

    def _phase(self, f, tick):
        if self.legacy:
            return self._state(f, tick).get('phase', 'UNKNOWN')
        return f['frames/phase'][tick].decode('ascii')

    def frames(self, start=0, stop=None):
        """Indices refer to the stored video, NOT 60-Hz ticks."""
        return iter_video(safe_relative(self.path, self.meta['rgb_file']),
                          self.settings['width'], self.settings['height'], start, stop)

    def _sample(self, f, index, before, after, privileged):
        a, b = map(int, self.rgb_tick_ids[index:index+2])
        seq = f['actions/value'][a:b]
        result = {
            'observation': before,
            'next_observation': after,
            'action': seq[0].copy(),
            'action_sequence': seq,
            'action_is_constant': bool(np.all(seq == seq[0])),
            'action_semantics': 'instantaneous_at_frame_id; transition uses entire action_sequence',
            'substeps': b-a,
            'seconds': (b-a) / self.settings['tick_hz'],
            'terminated': bool(f['actions/terminated'][b-1]),
            'truncated': bool(f['actions/truncated'][b-1]),
            'frame_id': a,
            'next_frame_id': b,
            'rgb_index': index,
            'phase': self._phase(f, a),
            'episode_id': self.meta['episode_id'],
        }
        if privileged:
            result['privileged'] = self._state(f, a)
            result['next_privileged'] = self._state(f, b)
        return result

    def _accept(self, f, index, phases):
        if phases is None:
            return True
        a, b = map(int, self.rgb_tick_ids[index:index+2])
        # Do not hide disallowed ticks inside a slower visual transition.
        return all(self._phase(f, t) in phases for t in range(a, b))

    def iter_transitions(self, privileged=False, phases=None):
        phases = _phase_filter(phases)
        video = self.frames()
        with h5py.File(self.path / 'data.h5', 'r') as f:
            try:
                before = next(video)
                for i in range(len(self)):
                    after = next(video)
                    if self._accept(f, i, phases):
                        yield self._sample(f, i, before, after, privileged)
                    before = after
            except StopIteration as e:
                raise IOError('Video incompleto') from e
            finally:
                video.close()

    def transition(self, t, privileged=False):
        """t is the RGB/video index. frame_id in the result is the native tick."""
        if not 0 <= t < len(self):
            raise IndexError(t)
        frames = list(self.frames(t, t+2))
        if len(frames) != 2:
            raise IOError('Video incompleto')
        with h5py.File(self.path / 'data.h5', 'r') as f:
            return self._sample(f, t, *frames, privileged)

    def iter_ticks(self, privileged=False):
        """Raw controls at tick_hz. No artificial RGB at unsaved instants."""
        with h5py.File(self.path / 'data.h5', 'r') as f:
            for t in range(self.num_ticks):
                result = {
                    'frame_id': t,
                    'action': f['actions/value'][t],
                    'phase': self._phase(f, t),
                    'terminated': bool(f['actions/terminated'][t]),
                    'truncated': bool(f['actions/truncated'][t]),
                }
                if privileged:
                    result['privileged'] = self._state(f, t)
                yield result

    @staticmethod
    def _block(chunk):
        result = {
            'observation': chunk[0]['observation'],
            'next_observation': chunk[-1]['next_observation'],
            'action_sequence': np.concatenate([x['action_sequence'] for x in chunk]),
            'substeps': sum(x['substeps'] for x in chunk),
            'seconds': sum(x['seconds'] for x in chunk),
            'frame_id': chunk[0]['frame_id'],
            'next_frame_id': chunk[-1]['next_frame_id'],
            'terminated': chunk[-1]['terminated'],
            'truncated': chunk[-1]['truncated'],
        }
        if 'privileged' in chunk[0]:
            result['privileged'] = chunk[0]['privileged']
            result['next_privileged'] = chunk[-1]['next_privileged']
        return result

    def iter_action_blocks(self, repeat=2, privileged=False):
        """Blocks of native ticks. Requires RGB at each block boundary."""
        period = (self.settings['tick_hz'] // self.settings.get('rgb_hz', self.settings['tick_hz'])
                  if not self.legacy else 1)
        if type(repeat) is not int or repeat < period or repeat % period:
            raise ValueError('repeat debe ser múltiplo del intervalo RGB; no se inventan frames intermedios')
        chunk, ticks = [], 0
        for sample in self.iter_transitions(privileged=privileged):
            chunk.append(sample)
            ticks += sample['substeps']
            if ticks >= repeat:
                yield self._block(chunk)
                chunk, ticks = [], 0
        if chunk:
            yield self._block(chunk)

    def iter_sequences(self, length=16, stride=16, privileged=False, phases=None):
        if type(length) is not int or type(stride) is not int or length < 1 or stride < 1:
            raise ValueError('length/stride deben ser enteros positivos')
        window = deque(maxlen=length)
        last, count = None, 0
        for sample in self.iter_transitions(privileged, phases):
            if last is not None and last != sample['frame_id']:
                window.clear()
                count = 0
            last = sample['next_frame_id']
            window.append(sample)
            count += 1
            if len(window) == length and (count-length) % stride == 0:
                rows = list(window)
                seq = [x['action_sequence'] for x in rows]
                substeps = np.asarray([len(x) for x in seq], np.int64)
                maxlen = int(substeps.max())
                padded = np.zeros((length, maxlen, 7), np.float32)
                mask = np.zeros((length, maxlen), bool)
                for i, actions in enumerate(seq):
                    padded[i, :len(actions)] = actions
                    mask[i, :len(actions)] = True
                result = {
                    'observations': np.stack([x['observation'] for x in rows]),
                    'actions': np.stack([x['action'] for x in rows]),
                    'action_sequences': padded,
                    'action_sequence_mask': mask,
                    'substeps': substeps,
                    'last_next_observation': rows[-1]['next_observation'],
                    'frame_start': rows[0]['frame_id'],
                    'frame_ids': np.array([x['frame_id'] for x in rows]),
                    'episode_id': self.meta['episode_id'],
                    'terminated': np.array([x['terminated'] for x in rows]),
                    'truncated': np.array([x['truncated'] for x in rows]),
                }
                if privileged:
                    result['privileged'] = [x['privileged'] for x in rows]
                yield result


def discover_episodes(root):
    root = Path(root)
    if (root / 'READY.json').exists():
        return [root]
    return sorted(p.parent for p in root.rglob('READY.json')
                  if not any(x.endswith(('.partial', '.uploading')) for x in p.parts))


class DemonstrationDataset:
    """Episode collection; phase filtering never stitches noncontiguous windows."""
    def __init__(self, root, allowed_sources=('human',), phases=CONTROL_PHASES):
        self.phases = _phase_filter(phases)
        candidates = discover_episodes(root) if isinstance(root, (str, Path)) else [Path(p) for p in root]
        self.episodes = []
        for path in candidates:
            ep = Episode(path)
            if allowed_sources is None or ep.meta.get('source') in allowed_sources:
                self.episodes.append(ep)
        if not self.episodes:
            raise ValueError('No hay episodios finalizados de las fuentes solicitadas')

    def __len__(self):
        if self.phases is None:
            return sum(map(len, self.episodes))
        n = 0
        for ep in self.episodes:
            with h5py.File(ep.path / 'data.h5', 'r') as f:
                n += sum(ep._accept(f, i, self.phases) for i in range(len(ep)))
        return n

    def _order(self, seed, shuffle, worker_id, num_workers):
        if num_workers < 1 or not 0 <= worker_id < num_workers:
            raise ValueError('worker_id/num_workers inválidos')
        order = np.arange(len(self.episodes))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        return order[worker_id::num_workers]

    def iter_transitions(self, *, privileged=False, shuffle_episodes=False,
                         seed=0, worker_id=0, num_workers=1):
        for i in self._order(seed, shuffle_episodes, worker_id, num_workers):
            yield from self.episodes[i].iter_transitions(privileged, self.phases)

    def iter_sequences(self, length=16, stride=16, *, privileged=False,
                       shuffle_episodes=False, seed=0, worker_id=0, num_workers=1):
        for i in self._order(seed, shuffle_episodes, worker_id, num_workers):
            yield from self.episodes[i].iter_sequences(length, stride, privileged, self.phases)

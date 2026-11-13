from __future__ import annotations
from pathlib import Path
import collections
import h5py
import numpy as np
from .dataset import Episode, discover_episodes
from .recording import validate_episode
from .types import ACTION_NAMES


def audit_episode(path, deep=False):
    path = Path(path)
    integrity = validate_episode(path, verify_hashes=True, decode_video=deep)
    if not integrity['valid']:
        return integrity
    ep = Episode(path, validate=False)
    meta = ep.meta
    t = ep.num_ticks
    files = {
        f.name: {'bytes': f.stat().st_size, 'MiB': round(f.stat().st_size / 2**20, 3)}
        for f in path.iterdir() if f.is_file()
    }
    datasets = {}
    groups = collections.Counter()
    nonzero_counts = {}
    with h5py.File(path / 'data.h5', 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                physical = int(obj.id.get_storage_size())
                datasets[name] = {
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'logical_bytes': int(obj.size * obj.dtype.itemsize),
                    'stored_bytes': physical,
                }
                groups[name.split('/')[0]] += physical
        f.visititems(visit)
        actions = f['actions/value'][:]
        stats = {}
        for i, name in enumerate(ACTION_NAMES):
            values = actions[:, i]
            nonzero = int(np.count_nonzero(values))
            nonzero_counts[name] = nonzero
            row = {
                'min': float(values.min()),
                'max': float(values.max()),
                'nonzero_fraction': float(np.mean(values != 0)),
                'nonzero_steps': nonzero,
                'changed_steps': int(np.count_nonzero(np.diff(values))),
            }
            if i >= 2:
                row['rising_edges'] = int(np.count_nonzero(np.diff(np.r_[0, values]) > 0))
                row['note'] = 'Button transitions, NOT verified game events'
            stats[name] = row
        phases = collections.Counter(ep._phase(f, i) for i in range(t))
        action_block_changes = sum(
            not np.all(actions[int(lo):int(hi)] == actions[int(lo)])
            for lo, hi in zip(ep.rgb_tick_ids[:-1], ep.rgb_tick_ids[1:])
        )
    total = sum(x['bytes'] for x in files.values())
    seconds = float(meta['simulated_seconds'])
    missing = [name for name in ACTION_NAMES if nonzero_counts[name] == 0]
    warnings = []
    if missing:
        warnings.append('Acciones nunca activadas en este episodio: ' + ', '.join(missing))
    return {
        'valid': True,
        'integrity': integrity,
        'episode_id': meta['episode_id'],
        'schema_version': meta['schema_version'],
        'files': files,
        'total_MiB': round(total / 2**20, 3),
        'estimated_GiB_per_hour_at_this_episode_rate': total / seconds * 3600 / 2**30,
        'estimate_note': 'Extrapolación de ESTA carrera; escenas/circuitos/compresión pueden cambiar el resultado',
        'hdf5_stored_MiB_by_group': {k: round(v / 2**20, 3) for k, v in groups.items()},
        'hdf5_metadata_and_unassigned_bytes': max(0, (path / 'data.h5').stat().st_size - sum(groups.values())),
        'hdf5_datasets': datasets,
        'ticks': t,
        'state_frames': t + 1,
        'rgb_frames': len(ep.rgb_tick_ids),
        'visual_intervals_with_changing_controls': int(action_block_changes),
        'phase_ticks': dict(phases),
        'phase_seconds': {k: v / ep.settings['tick_hz'] for k, v in phases.items()},
        'actions': stats,
        'action_coverage': {
            'missing_nonzero_actions': missing,
            'all_actions_observed': not missing,
            'warning': warnings[0] if warnings else None,
        },
        'warnings': warnings,
        'quality': meta.get('quality', {}),
        'roster': meta.get('roster', []),
        'human_slot': meta.get('human_slot'),
        'effective_seed': meta.get('effective_seed', ep.settings.get('seed')),
        'note': 'Integridad y conteos no validan sincronización perceptual ni calidad de las demostraciones.',
    }


def audit_collection(root, deep=False):
    paths = discover_episodes(root)
    if not paths:
        raise ValueError('No se encontraron episodios finalizados (READY.json)')
    reports = [audit_episode(path, deep=deep) for path in paths]
    invalid = [r for r in reports if not r.get('valid')]
    action_nonzero = collections.Counter({name: 0 for name in ACTION_NAMES})
    action_rising = collections.Counter({name: 0 for name in ACTION_NAMES[2:]})
    total_bytes = 0
    seconds = 0.0
    seeds = set()
    slots = collections.Counter()
    episodes = []
    for path, r in zip(paths, reports):
        if not r.get('valid'):
            episodes.append({'path': str(path), 'valid': False, 'errors': r.get('errors', [])})
            continue
        ep = Episode(path, validate=False)
        total_bytes += sum(x['bytes'] for x in r['files'].values())
        seconds += float(ep.meta['simulated_seconds'])
        if r.get('effective_seed') is not None:
            seeds.add(int(r['effective_seed']))
        if r.get('human_slot') is not None:
            slots[int(r['human_slot'])] += 1
        for name in ACTION_NAMES:
            action_nonzero[name] += int(r['actions'][name]['nonzero_steps'])
            if name in action_rising:
                action_rising[name] += int(r['actions'][name].get('rising_edges', 0))
        episodes.append({
            'path': str(path), 'valid': True, 'episode_id': r['episode_id'],
            'seconds': float(ep.meta['simulated_seconds']), 'MiB': r['total_MiB'],
            'seed': r.get('effective_seed'), 'human_slot': r.get('human_slot'),
        })
    missing = [name for name in ACTION_NAMES if action_nonzero[name] == 0]
    warnings = []
    if missing:
        warnings.append('Acciones nunca observadas en TODO el conjunto: ' + ', '.join(missing))
    if len(seeds) < max(1, len(reports) - len(invalid)):
        warnings.append('Hay seeds repetidos entre episodios; revisa la diversidad de recopilación')
    return {
        'valid': not invalid,
        'scope': 'collection',
        'root': str(Path(root)),
        'episodes_found': len(paths),
        'episodes_valid': len(paths) - len(invalid),
        'episodes_invalid': len(invalid),
        'total_hours': seconds / 3600,
        'total_GiB': total_bytes / 2**30,
        'unique_effective_seeds': len(seeds),
        'human_slot_counts': dict(sorted(slots.items())),
        'action_coverage': {
            'nonzero_steps': dict(action_nonzero),
            'button_rising_edges': dict(action_rising),
            'missing_nonzero_actions': missing,
            'all_actions_observed': not missing,
        },
        'warnings': warnings,
        'episodes': episodes,
    }


def audit_path(path, deep=False):
    path = Path(path)
    if (path / 'READY.json').exists():
        return audit_episode(path, deep=deep)
    return audit_collection(path, deep=deep)

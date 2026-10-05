"""Independent MPS screen of PEMS-BAY STGCN candidates; validation only.

The METR-LA CUDA search and the stopped local studies are separate protocols.
"""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from traffic_forecasting import multihorizon_mps as mps
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.multihorizon_data import MaskedWindows, atomic_json, load_development
from traffic_forecasting.search_plan import validate_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing a silent CPU fallback')
    protocol = json.loads(args.protocol.read_text())
    if protocol['datasets'] != ['PEMS_BAY'] or protocol['seeds'] != [17]:
        raise ValueError('This screen is frozen to PEMS-BAY seed 17')
    if any(t['model'] != 'STGCN' for t in protocol['trials']):
        raise ValueError('Only STGCN trials are allowed')
    for trial in protocol['trials']:
        validate_trial(trial)
    torch.set_num_threads(protocol['threads'])
    torch.use_deterministic_algorithms(True)
    mps.install()
    study.CODE_FILES.extend(('traffic_forecasting/local_pems_stgcn_mps.py',
                             'traffic_forecasting/search_plan.py'))
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        declaration = study.freeze(root, protocol)
        values, graph, _, manifest = load_development(args.data_dir, 'PEMS_BAY')
        dataset_root = root / 'PEMS_BAY'
        dataset_root.mkdir(exist_ok=True)
        manifest_path = dataset_root / 'data_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Data manifest changed since freeze')
        atomic_json(manifest_path, manifest)
        train_end, val_end = manifest['boundaries']
        train_windows = MaskedWindows(values, 0, train_end, manifest['mean'], manifest['std'])
        val_windows = MaskedWindows(values, train_end, val_end, manifest['mean'], manifest['std'])
        results = []
        for trial in protocol['trials']:
            batch_size = trial['batch_size']
            train_loader = mps.MPSDataLoader(train_windows, batch_size=batch_size,
                                             shuffle=True, num_workers=0)
            val_loader = mps.MPSDataLoader(val_windows, batch_size=batch_size,
                                           shuffle=False, num_workers=0)
            seed = 17
            out = dataset_root / trial['id'] / str(seed)
            identity = study.fingerprint({'declaration': declaration, 'data': manifest,
                                          'trial': trial, 'seed': seed})
            result = study.train_trial(trial, seed, graph, train_loader, val_loader,
                                       manifest, protocol, out, identity)
            results.append(result)
            atomic_json(dataset_root / 'screen_progress.json', {
                'complete': len(results), 'total': len(protocol['trials']),
                'validation_only': True,
                'results': [{'trial': x['trial'],
                             'validation_mae_mph': x['validation']['all']['overall']['mae_mph']}
                            for x in results]})
        order = sorted(results, key=lambda x: (x['validation']['all']['overall']['mae_mph'],
                                                x['trial']))
        atomic_json(dataset_root / 'screen_ranking.json', {
            'metric': protocol['selection'], 'test_status': 'not evaluated',
            'ranked': [{'trial': x['trial'], 'validation_mae_mph':
                        x['validation']['all']['overall']['mae_mph']} for x in order]})


if __name__ == '__main__':
    main()

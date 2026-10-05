"""Independent MPS screen of PEMS-BAY STGCN candidates; validation only.

The METR-LA CUDA search and the stopped local studies are separate protocols.
"""
import argparse
import fcntl
import json
import time
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
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing a silent CPU fallback')
    protocol = json.loads(args.protocol.read_text())
    if protocol['datasets'] != ['PEMS_BAY'] or protocol['seeds'] != [42, 73]:
        raise ValueError('This confirmation is frozen to PEMS-BAY seeds42/73')
    if any(t['model'] != 'STGCN' for t in protocol['trials']):
        raise ValueError('Only STGCN trials are allowed')
    for trial in protocol['trials']:
        validate_trial(trial)
    torch.set_num_threads(protocol['threads'])
    torch.use_deterministic_algorithms(True)
    mps.install()
    study.CODE_FILES.extend(('traffic_forecasting/local_pems_stgcn_mps_confirm.py',
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
        if args.smoke_only:
            checks = []
            for trial in protocol['trials']:
                batch = next(iter(mps.MPSDataLoader(train_windows, batch_size=trial['batch_size'])))
                model = study.build_model(trial, graph)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=trial['learning_rate'])
                pred = study.predict(model, batch, training=True, step=0)
                assert pred.shape == batch['y'].shape
                loss = study.masked_loss(pred, batch['y'], batch['y_mask'], manifest['std'])
                loss.backward()
                assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
                optimizer.step()
                model.eval()
                with torch.inference_mode():
                    assert torch.isfinite(study.predict(model, batch)).all()
                checks.append({'trial':trial['id'], 'batch_size':trial['batch_size'], 'passed':True})
            atomic_json(root / 'smoke.json', {'device':'mps', 'checks':checks})
            return
        started = time.monotonic()
        class BudgetLoader(mps.MPSDataLoader):
            def __iter__(self):
                if time.monotonic() - started >= 18 * 3600:
                    raise RuntimeError('Wall budget reached at epoch boundary; resume checkpoint preserved')
                return super().__iter__()
        trials = {t['id']:t for t in protocol['trials']}
        results = []
        for task in protocol['tasks']:
            trial, seed = trials[task['trial']], task['seed']
            train_loader = BudgetLoader(train_windows, batch_size=trial['batch_size'], shuffle=True, num_workers=0)
            val_loader = mps.MPSDataLoader(val_windows, batch_size=trial['batch_size'], shuffle=False, num_workers=0)
            out = dataset_root / trial['id'] / str(seed)
            identity = study.fingerprint({'declaration':declaration, 'data':manifest, 'trial':trial, 'seed':seed})
            result = study.train_trial(trial, seed, graph, train_loader, val_loader, manifest, protocol, out, identity)
            results.append(result)
            atomic_json(root / 'confirmation_progress.json', {'complete':len(results), 'total':len(protocol['tasks']), 'validation_only':True, 'results':results})


if __name__ == '__main__':
    main()

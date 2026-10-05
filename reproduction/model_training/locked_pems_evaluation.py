"""Freeze complete validation selection, then evaluate the held-out PEMS tail once."""
import argparse
import fcntl
import hashlib
import json
import os
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'Bigscity-LibCity-master'
sys.path.insert(0, str(REPO))
import torch
from training_safety import atomic_record
from validate_training_artifacts import validate
from traffic_forecasting import multihorizon_mps as mps
from traffic_forecasting.multihorizon_data import load_development, MaskedWindows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    rows = []
    manifests = set()
    numerical_protocols = set()
    environments = set()
    configs = {}
    for trial in ['stgcn_reference', 'stgcn_search_005']:
        for seed in [17, 42, 73]:
            cohort = ('local_pems_stgcn_mps_screen_v1' if seed == 17 else
                      'local_pems_stgcn_mps_seed73_v1' if trial == 'stgcn_search_005' and seed == 73
                      else 'local_pems_stgcn_mps_confirm_v1')
            directory = ROOT / 'runs' / cohort / 'PEMS_BAY' / trial / str(seed)
            audit = validate(directory)
            declaration = json.loads((directory.parents[2] / 'frozen_protocol.json').read_text())
            numerical_protocols.add(json.dumps({k: declaration['protocol'][k] for k in ['epochs','min_epochs','patience','threads','hardware','data']}, sort_keys=True))
            environments.add(json.dumps(declaration['environment'], sort_keys=True))
            checkpoint_config = next(t for t in declaration['protocol']['trials'] if t['id'] == trial)
            if trial in configs and configs[trial] != checkpoint_config:
                raise ValueError('Configuration differs across matched seeds')
            configs[trial] = checkpoint_config
            for name, expected in declaration['code_sha256'].items():
                if digest(REPO / name) != expected:
                    raise ValueError('Frozen training source changed: ' + name)
            manifests.add(digest(directory.parents[1] / 'data_manifest.json'))
            result = json.loads((directory / 'result.json').read_text())
            rows.append({'trial': trial, 'seed': seed, 'directory': str(directory.relative_to(ROOT)),
                         'identity': result['identity'], 'hashes': audit['hashes'],
                         'validation_mae_mph': result['validation']['all']['overall']['mae_mph']})
    if len(numerical_protocols) != 1 or len(environments) != 1:
        raise ValueError('Numerical protocol or environment mismatch')
    if len(manifests) != 1:
        raise ValueError('Different data manifests cannot be pooled')
    return rows, manifests.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    os.umask(0o077)
    output = ROOT / 'runs/final_pems_evaluation_v1'
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads((ROOT / 'test_provenance_ledger.json').read_text())
        if ledger.get('status') != 'reviewed_independent_with_disclosed_scope':
            raise ValueError('Test provenance review is required')
        rows, manifest = collect()
        means = {trial: statistics.mean(r['validation_mae_mph'] for r in rows if r['trial'] == trial)
                 for trial in ['stgcn_reference', 'stgcn_search_005']}
        selection = {'version': 'final_pems_evaluation_v1', 'models': rows,
                     'winner_by_validation': min(means, key=means.get), 'validation_means': means,
                     'primary_metric': 'Equal mean of 12 horizon MAEs in mph',
                     'data_manifest_sha256': manifest, 'evaluation_source_sha256': digest(Path(__file__)),
                     'provenance_ledger_sha256': digest(ROOT / 'test_provenance_ledger.json'),
                     'device': 'Apple M4 Pro MPS; no CPU fallback',
                     'test_partition': 'Last 20%, partition-contained 12-to-12 windows',
                     'baselines': ['persistence', 'moving_mean'],
                     'representative_serving_seed': 42,
                     'limits': ['Seed17 participated in screening', 'No selection based on test scores']}
        path = output / 'locked_selection.json'
        if args.mode == 'freeze':
            if path.exists():
                if json.loads(path.read_text()) != selection:
                    raise ValueError('Refusing to replace locked selection')
            else:
                atomic_record(path, selection)
            print('Selection frozen. No test predictions generated.')
            return
        if not path.exists() or json.loads(path.read_text()) != selection:
            raise ValueError('Freeze complete selection before test evaluation')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no fallback')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        values, graph, _, data = load_development(ROOT / 'data', 'PEMS_BAY')
        frozen_data = json.loads((ROOT / rows[0]['directory']).parents[1].joinpath('data_manifest.json').read_text())
        if data != frozen_data:
            raise ValueError('Data, split or normalization changed')
        windows = MaskedWindows(values, data['boundaries'][1], len(values), data['mean'], data['std'])
        loader = mps.MPSDataLoader(windows, batch_size=16, shuffle=False, num_workers=0)
        for row in rows:
            report = output / f"{row['trial']}_seed{row['seed']}.json"
            if report.exists():
                prior = json.loads(report.read_text())
                if prior['selection_sha256'] != digest(path):
                    raise ValueError('Cached evaluation belongs to another selection')
                continue
            checkpoint = torch.load(ROOT / row['directory'] / 'selected.pt', map_location='cpu', weights_only=True)
            model = mps.build_model(checkpoint['trial'], graph)
            model.load_state_dict(checkpoint['state_dict'])
            metrics = mps.score(model, loader, data['std'], data['mean'], diagnostics=True)
            atomic_record(report, {'selection_sha256': digest(path), 'trial': row['trial'], 'seed': row['seed'],
                                  'test': metrics, 'primary_mae_mph': statistics.mean(
                                      h['mae_mph'] for h in metrics['all']['horizons'].values())})
            del model
        for baseline in selection['baselines']:
            report = output / f'baseline_{baseline}.json'
            if report.exists():
                prior = json.loads(report.read_text())
                if prior.get('selection_sha256') != digest(path):
                    raise ValueError('Cached baseline belongs to another selection')
                if prior.get('baseline') != baseline:
                    raise ValueError('Cached baseline identity mismatch')
            else:
                metrics = mps.score(None, loader, data['std'], data['mean'], diagnostics=True, baseline=baseline)
                atomic_record(report, {'selection_sha256': digest(path), 'baseline': baseline, 'test': metrics})
        print('Locked evaluation complete; never retune against these test scores.')


if __name__ == '__main__':
    main()

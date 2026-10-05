"""Independent nonlinear temporal baseline for 12-to-12 speed forecasting.

This is a sensor-shared MLP, not a replacement for the frozen CARC graph search.
Only training and validation partitions are loaded into windows.
"""
import argparse
import fcntl
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from libcity.model.forecasting_utils import last_observed_value
from traffic_forecasting.local_linear import batches, masked_mae, score, sync
from traffic_forecasting.multihorizon_data import atomic_json, load_development, sha256


class TemporalMLP(nn.Module):
    """Per-sensor nonlinear correction with an explicit observed-value mask."""

    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, observed):
        if x.ndim != 4 or x.shape[1] != 12 or x.shape[-1] != 1 or observed.shape != x.shape:
            raise ValueError('Expected matching [batch, 12, sensors, 1] tensors')
        anchor = last_observed_value(x, observed)
        centered = torch.where(observed, x - anchor, 0).squeeze(-1).transpose(1, 2)
        features = torch.cat((centered, observed.squeeze(-1).transpose(1, 2).float()), dim=-1)
        return self.net(features).transpose(1, 2).unsqueeze(-1) + anchor


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def epoch(model, optimizer, values, bounds, mean, std, batch_size, device, seed=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    for x, observed, y, valid in batches(values, bounds, mean, std, batch_size, device, seed):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x, observed)
        loss = masked_mae(pred, y, valid, std)
        if loss is None:
            continue
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        n = int(valid.sum().item())
        total += float(loss.detach()) * n
        count += n
    if count == 0:
        raise ValueError('No observed targets in training epoch')
    sync(device)
    return total / count


def preflight(values, manifest, protocol, device):
    """Measure a full train plus validation epoch; smoke gradients and inference."""
    torch.manual_seed(1337)
    model = TemporalMLP(protocol['width']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    started = time.monotonic()
    epoch(model, optimizer, values, (0, manifest['boundaries'][0]), manifest['mean'], manifest['std'],
          protocol['batch_size'], device, 1337)
    train_seconds = time.monotonic() - started
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Missing or nonfinite gradients')
    started = time.monotonic()
    validation = score(model, values, tuple(manifest['boundaries']), manifest['mean'], manifest['std'],
                       protocol['batch_size'], device)
    validation_seconds = time.monotonic() - started
    return {'device': device.type, 'train_seconds': train_seconds,
            'validation_seconds': validation_seconds, 'total_seconds': train_seconds + validation_seconds,
            'validation_mae_mph': validation['selection_mae_mph'], 'gradient_check': 'passed'}


def train(root, dataset, values, manifest, protocol, device, seed):
    out = root / dataset / 'temporal_mlp' / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'protocol': protocol, 'data_hashes': manifest['hashes'],
                'dataset': dataset, 'seed': seed, 'device': device.type}
    config = out / 'effective_config.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError('Frozen trial identity changed')
    atomic_json(config, identity)
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TemporalMLP(protocol['width']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'])
    history, best, stale = [], None, 0
    progress = out / 'progress.pt'
    if progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best, stale = state['history'], state['best'], state['stale']
    for index in range(len(history), protocol['epochs']):
        if index >= protocol['min_epochs'] and stale >= protocol['patience']:
            break
        started = time.monotonic()
        train_mae = epoch(model, optimizer, values, (0, manifest['boundaries'][0]),
                          manifest['mean'], manifest['std'], protocol['batch_size'], device,
                          seed + index * 100003)
        validation = score(model, values, tuple(manifest['boundaries']),
                           manifest['mean'], manifest['std'], protocol['batch_size'], device)
        if best is None or validation['selection_mae_mph'] < best['validation']['selection_mae_mph']:
            best = {'epoch': index + 1, 'validation': validation,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        row = {'epoch': index + 1, 'train_mae_mph': train_mae,
               'validation': validation, 'duration_s': time.monotonic() - started}
        history.append(row)
        save_checkpoint(progress, {'identity': identity,
                        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(), 'history': history,
                        'best': best, 'stale': stale})
        atomic_json(out / 'epochs.json', history)
        print(json.dumps({'dataset': dataset, 'seed': seed, 'epoch': index + 1,
                          'validation_mae_mph': validation['selection_mae_mph'],
                          'duration_s': row['duration_s']}), flush=True)
    save_checkpoint(out / 'selected.pt', {'identity': identity, **best})
    result = {'dataset': dataset, 'model': 'temporal_mlp', 'seed': seed,
              'device': device.type, 'epochs': len(history), 'selected_epoch': best['epoch'],
              'validation': best['validation'], 'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': sum(row['duration_s'] for row in history)}
    atomic_json(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=['METR_LA', 'PEMS_BAY'],
                        default=['METR_LA', 'PEMS_BAY'])
    args = parser.parse_args()
    protocol = {'version': 'local_temporal_mlp_m4pro_v1', 'width': 64,
                'seeds': [17, 42, 73], 'epochs': 80, 'min_epochs': 15,
                'patience': 10, 'batch_size': 256, 'learning_rate': .001,
                'weight_decay': 0., 'history': 12, 'horizon': 12,
                'selection': 'equal-weight mean of 12 validation horizon MAEs; test untouched'}
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repo = Path(__file__).resolve().parents[1]
        frozen = {'protocol': protocol, 'source_sha256': {name: sha256(repo / name) for name in
                  ('traffic_forecasting/local_mlp.py', 'traffic_forecasting/local_linear.py',
                   'traffic_forecasting/multihorizon_data.py', 'libcity/model/forecasting_utils.py')},
                  'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                  'numpy': np.__version__, 'machine': platform.machine()}}
        frozen_path = root / 'frozen_protocol.json'
        if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
            raise ValueError('Frozen source, protocol, or environment changed')
        atomic_json(frozen_path, frozen)
        for dataset in args.datasets:
            values, _, _, manifest = load_development(args.data_dir, dataset)
            out = root / dataset
            out.mkdir(exist_ok=True)
            manifest_path = out / 'data_manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Data manifest changed')
            atomic_json(manifest_path, manifest)
            device_path = out / 'device_selection.json'
            if device_path.exists():
                selection = json.loads(device_path.read_text())
            else:
                measured = [preflight(values, manifest, protocol, torch.device('cpu'))]
                if torch.backends.mps.is_available():
                    measured.append(preflight(values, manifest, protocol, torch.device('mps')))
                selection = {'chosen': min(measured, key=lambda item: item['total_seconds'])['device'],
                             'criterion': 'complete training and validation epoch', 'measured': measured}
                atomic_json(device_path, selection)
            if selection['chosen'] == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError('Frozen MPS study cannot silently fall back to CPU')
            device = torch.device(selection['chosen'])
            results = [train(root, dataset, values, manifest, protocol, device, seed)
                       for seed in protocol['seeds']]
            metrics = [row['validation']['selection_mae_mph'] for row in results]
            atomic_json(out / 'validation_summary.json', {
                'results': results, 'mean_mae_mph': float(np.mean(metrics)),
                'sample_sd_mae_mph': float(np.std(metrics, ddof=1)),
                'test_status': 'not evaluated'})


if __name__ == '__main__':
    main()

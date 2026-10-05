"""Independent, validation-only 12-to-12 linear baselines for Apple silicon.

This study does not import or modify the frozen CPU, MPS, or CUDA trainers.
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
from torch.nn import functional as F

from libcity.model.forecasting_utils import last_observed_value
from traffic_forecasting.multihorizon_data import atomic_json, load_development, sha256


class LinearForecast(nn.Module):
    """A sensor-shared linear correction to the latest observed speed."""

    def __init__(self, kind):
        super().__init__()
        if kind not in ('linear', 'dlinear'):
            raise ValueError(kind)
        self.kind = kind
        self.linear = nn.Linear(12, 12)
        if kind == 'dlinear':
            self.trend = nn.Linear(12, 12)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if kind == 'dlinear':
            nn.init.zeros_(self.trend.weight)
            nn.init.zeros_(self.trend.bias)

    def forward(self, x, observed):
        if x.ndim != 4 or x.shape[1] != 12 or x.shape[-1] != 1 or observed.shape != x.shape:
            raise ValueError('Expected matching [batch, 12, sensors, 1] tensors')
        anchor = last_observed_value(x, observed)
        centered = torch.where(observed, x - anchor, 0).squeeze(-1).transpose(1, 2)
        if self.kind == 'linear':
            correction = self.linear(centered)
        else:
            smooth = F.avg_pool1d(F.pad(centered.reshape(-1, 1, 12), (1, 1), mode='replicate'), 3, 1)
            trend = smooth.reshape_as(centered)
            correction = self.linear(centered - trend) + self.trend(trend)
        return correction.transpose(1, 2).unsqueeze(-1) + anchor


def make_batch(values, starts, mean, std, device):
    offsets = np.arange(24)
    raw = np.asarray(values[starts[:, None] + offsets[None, :]])
    mask = np.isfinite(raw) & (raw > 0)
    normalized = np.where(mask, (raw - mean) / std, 0).astype(np.float32)
    data = torch.from_numpy(normalized).to(device)
    observed = torch.from_numpy(mask).to(device)
    return data[:, :12, :, None], observed[:, :12, :, None], data[:, 12:, :, None], observed[:, 12:, :, None]


def batches(values, bounds, mean, std, batch_size, device, seed=None):
    lo, hi = bounds
    starts = np.arange(lo, hi - 23)
    if seed is not None:
        starts = np.random.default_rng(seed).permutation(starts)
    for offset in range(0, len(starts), batch_size):
        yield make_batch(values, starts[offset:offset + batch_size], mean, std, device)


def masked_mae(pred, y, mask, std):
    if pred.shape != y.shape or mask.shape != y.shape:
        raise ValueError('Prediction, target, and mask shapes disagree')
    return (pred - y).abs()[mask].mean() * std if mask.any() else None


@torch.inference_mode()
def score(model, values, bounds, mean, std, batch_size, device, baseline=None):
    if model is not None:
        model.eval()
    absolute = np.zeros(12, np.float64)
    squared = np.zeros(12, np.float64)
    count = np.zeros(12, np.float64)
    for x, observed, y, valid in batches(values, bounds, mean, std, batch_size, device):
        if baseline == 'persistence':
            pred = last_observed_value(x, observed).expand_as(y)
        elif baseline == 'moving_mean':
            pred = ((x * observed).sum(1, keepdim=True) /
                    observed.sum(1, keepdim=True).clamp_min(1)).expand_as(y)
        else:
            pred = model(x, observed)
        if not torch.isfinite(pred).all():
            raise ValueError('Nonfinite validation prediction')
        error = ((pred - y) * std).detach().cpu().double().numpy()[..., 0]
        mask = valid.detach().cpu().numpy()[..., 0]
        absolute += (np.abs(error) * mask).sum(axis=(0, 2))
        squared += (error ** 2 * mask).sum(axis=(0, 2))
        count += mask.sum(axis=(0, 2))
    if np.any(count == 0):
        raise ValueError('A validation horizon has no observed targets')
    horizon_mae = absolute / count
    return {'selection_mae_mph': float(horizon_mae.mean()),
            'pooled_mae_mph': float(absolute.sum() / count.sum()),
            'pooled_rmse_mph': float(np.sqrt(squared.sum() / count.sum())),
            'horizon_mae_mph': {str(5 * (i + 1)): float(v) for i, v in enumerate(horizon_mae)},
            'observations': int(count.sum())}


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def benchmark(values, bounds, mean, std, kind, batch_size, device):
    """Time whole forward/backward/optimizer steps including batch transfer."""
    torch.manual_seed(719)
    model = LinearForecast(kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    starts = np.arange(bounds[0], min(bounds[1] - 23, bounds[0] + batch_size * 8))
    times = []
    for index in range(8):
        begin = time.perf_counter()
        x, observed, y, valid = make_batch(values, starts[index * batch_size:(index + 1) * batch_size],
                                            mean, std, device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_mae(model(x, observed), y, valid, std)
        if loss is None or not torch.isfinite(loss):
            raise ValueError('Invalid benchmark loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Invalid benchmark gradients')
        optimizer.step()
        sync(device)
        times.append(time.perf_counter() - begin)
    return {'device': device.type, 'kind': kind, 'batch_size': batch_size,
            'median_step_seconds': float(np.median(times[3:])), 'full_batch_checked': True}


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_one(root, dataset, kind, seed, values, manifest, protocol, device):
    target = root / dataset / kind / str(seed)
    target.mkdir(parents=True, exist_ok=True)
    identity = {'protocol': protocol, 'hashes': manifest['hashes'], 'dataset': dataset,
                'model': kind, 'seed': seed, 'device': device.type}
    config = target / 'effective_config.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError(f'Frozen trial identity changed: {target}')
    atomic_json(config, identity)
    if (target / 'result.json').exists():
        return json.loads((target / 'result.json').read_text())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = LinearForecast(kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'])
    history, best, stale = [], None, 0
    progress = target / 'progress.pt'
    if progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint belongs to another trial')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best, stale = state['history'], state['best'], state['stale']
    train_bounds = (0, manifest['boundaries'][0])
    val_bounds = tuple(manifest['boundaries'])
    for epoch in range(len(history), protocol['epochs']):
        if epoch >= protocol['min_epochs'] and stale >= protocol['patience']:
            break
        started = time.monotonic()
        model.train()
        total, count = 0., 0
        for x, observed, y, valid in batches(values, train_bounds, manifest['mean'], manifest['std'],
                                             protocol['batch_size'], device, seed + epoch * 100003):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(model(x, observed), y, valid, manifest['std'])
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            n = int(valid.sum().item())
            total += float(loss.detach()) * n; count += n
        validation = score(model, values, val_bounds, manifest['mean'], manifest['std'],
                           protocol['batch_size'], device)
        metric = validation['selection_mae_mph']
        if best is None or metric < best['validation']['selection_mae_mph']:
            best = {'epoch': epoch + 1, 'validation': validation,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        sync(device)
        row = {'epoch': epoch + 1, 'train_mae_mph': total / count,
               'validation': validation, 'duration_s': time.monotonic() - started}
        history.append(row)
        save_checkpoint(progress, {'identity': identity,
                        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(), 'history': history, 'best': best, 'stale': stale})
        atomic_json(target / 'epochs.json', history)
        print(json.dumps({'dataset': dataset, 'model': kind, 'seed': seed,
                          'epoch': epoch + 1, 'validation_mae_mph': metric,
                          'duration_s': row['duration_s']}), flush=True)
    save_checkpoint(target / 'selected.pt', {'identity': identity, **best})
    result = {'dataset': dataset, 'model': kind, 'seed': seed, 'device': device.type,
              'epochs': len(history), 'selected_epoch': best['epoch'],
              'validation': best['validation'], 'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': sum(v['duration_s'] for v in history)}
    atomic_json(target / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', choices=['METR_LA', 'PEMS_BAY'], default=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--mode', choices=['benchmark', 'train'], default='train')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    protocol = {'version': 'local_linear_m4pro_v1', 'models': ['linear', 'dlinear'],
                'seeds': [17, 42, 73], 'epochs': 80, 'min_epochs': 15, 'patience': 10,
                'batch_size': 256, 'learning_rate': .001, 'weight_decay': 0.,
                'selection': 'equal-weight mean of 12 validation horizon MAEs; test untouched',
                'history': 12, 'horizon': 12, 'precision': 'float32'}
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        declaration = {'protocol': protocol, 'source_sha256': {
            name: sha256(Path(__file__).resolve().parents[1] / name)
            for name in ['traffic_forecasting/local_linear.py', 'traffic_forecasting/multihorizon_data.py',
                         'libcity/model/forecasting_utils.py']},
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'numpy': np.__version__, 'machine': platform.machine()}}
        frozen = root / 'frozen_protocol.json'
        if frozen.exists() and json.loads(frozen.read_text()) != declaration:
            raise ValueError('Protocol or environment changed; use a new output directory')
        atomic_json(frozen, declaration)
        for dataset in args.datasets:
            values, _, _, manifest = load_development(args.data_dir, dataset)
            target = root / dataset
            target.mkdir(exist_ok=True)
            manifest_path = target / 'data_manifest.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Data manifest changed')
            atomic_json(manifest_path, manifest)
            train_bounds = (0, manifest['boundaries'][0])
            selected_path = target / 'device_selection.json'
            if selected_path.exists():
                selection = json.loads(selected_path.read_text())
                if selection['request'] != args.device:
                    raise ValueError('Device request changed within a frozen study')
                benchmark_results = json.loads((target / 'device_benchmark.json').read_text())
            else:
                benchmark_results = []
                for kind in protocol['models']:
                    for device_name in (['cpu', 'mps'] if args.device == 'auto' else [args.device]):
                        if device_name == 'mps' and not torch.backends.mps.is_available():
                            continue
                        benchmark_results.append(benchmark(values, train_bounds, manifest['mean'], manifest['std'],
                                                           kind, protocol['batch_size'], torch.device(device_name)))
                if not benchmark_results:
                    raise RuntimeError('No available device passed preflight')
                choice = {kind: min((item for item in benchmark_results if item['kind'] == kind),
                                    key=lambda item: item['median_step_seconds'])['device']
                          for kind in protocol['models']}
                selection = {'request': args.device, 'choice': choice,
                             'criterion': 'median full-batch training step, including transfer'}
                atomic_json(target / 'device_benchmark.json', benchmark_results)
                atomic_json(selected_path, selection)
            if args.mode == 'benchmark':
                print(json.dumps({'dataset': dataset, 'benchmarks': benchmark_results}), flush=True)
                continue
            validation_bounds = tuple(manifest['boundaries'])
            baselines = {kind: score(None, values, validation_bounds, manifest['mean'], manifest['std'],
                                     protocol['batch_size'], torch.device('cpu'), baseline=kind)
                         for kind in ('persistence', 'moving_mean')}
            atomic_json(target / 'validation_baselines.json', baselines)
            results = []
            for kind in protocol['models']:
                chosen = selection['choice'][kind]
                if chosen == 'mps' and not torch.backends.mps.is_available():
                    raise RuntimeError('Frozen MPS study cannot silently fall back to CPU')
                for seed in protocol['seeds']:
                    results.append(train_one(root, dataset, kind, seed, values, manifest,
                                             protocol, torch.device(chosen)))
            aggregate = {kind: {'mean': float(np.mean([r['validation']['selection_mae_mph'] for r in results
                                                       if r['model'] == kind])),
                                'sd': float(np.std([r['validation']['selection_mae_mph'] for r in results
                                                    if r['model'] == kind], ddof=1))}
                         for kind in protocol['models']}
            atomic_json(target / 'validation_summary.json', {'results': results, 'aggregate': aggregate,
                       'baselines': baselines, 'test_status': 'not evaluated'})


if __name__ == '__main__':
    main()

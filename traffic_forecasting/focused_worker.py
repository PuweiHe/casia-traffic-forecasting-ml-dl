"""One immutable validation-only task per process, with a per-task writer lock."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
import torch
from traffic_forecasting import cuda_search as search
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting import multihorizon_mps as placement
from traffic_forecasting.focused_models import SpeedSTAEformer
from traffic_forecasting.multihorizon_data import MaskedWindows, load_development, atomic_json


def install():
    search.install()
    original = study.build_model
    def build(trial, graph):
        if trial['model'] == 'STAEformer':
            return SpeedSTAEformer(trial['config'], len(graph)).to(placement.DEVICE)
        return original(trial, graph)
    study.build_model = build
    study.CODE_FILES.extend(['traffic_forecasting/focused_worker.py',
                            'traffic_forecasting/focused_models.py',
                            'libcity/model/traffic_speed_prediction/STAEformer.py'])


def run(args):
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.index < len(plan['tasks']):
        raise ValueError('Task index outside frozen manifest')
    task = plan['tasks'][args.index]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    install()
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    hardware = {'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                'memory': torch.cuda.get_device_properties(0).total_memory,
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
                'driver': subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    with (root / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        declaration = study.freeze(root, {'plan': plan, 'hardware': hardware, 'preflight': args.preflight})
    trial, seed = task['trial'], task['seed']
    out = root / trial['id'] / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        values, graph, _, data = load_development(args.data_dir, plan['dataset'])
        data['test_status'] = 'exploratory' if plan['dataset'] == 'METR_LA' else 'provenance_review_required'
        with (root / '.manifest.lock').open('a') as setup_lock:
            fcntl.flock(setup_lock, fcntl.LOCK_EX)
            search.immutable_json(root / 'data_manifest.json', data)
        ends = data['boundaries']
        windows = {k: MaskedWindows(values, lo, hi, data['mean'], data['std']) for k,(lo,hi) in
                   {'train':(0,ends[0]), 'validation':(ends[0],ends[1])}.items()}
        identity = study.fingerprint({'declaration':declaration,'manifest':data,'trial':trial,'seed':seed})
        search.immutable_json(out / 'effective_config.json', {'trial':trial,'seed':seed,'identity':identity})
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            if not (out / 'smoke.json').exists():
                atomic_json(out / 'smoke.json', search.smoke(trial, graph, windows['train'], data))
            loaders = {k: study.DataLoader(v,batch_size=trial['batch_size'],shuffle=k=='train',num_workers=0) for k,v in windows.items()}
            budget = dict(plan)
            if args.preflight:
                budget.update(epochs=3,min_epochs=3,patience=3)
            result = study.train_trial(trial,seed,graph,loaders['train'],loaders['validation'],data,budget,out,identity)
            atomic_json(out / 'segment_resources.json', {'wall_s':time.monotonic()-started,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(),'hardware':hardware,
                'preflight_only':args.preflight,'completed_epochs':result['completed_epochs']})
        except Exception:
            (out / 'error.log').write_text(traceback.format_exc())
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--preflight',action='store_true')
    run(p.parse_args())

if __name__=='__main__':
    main()

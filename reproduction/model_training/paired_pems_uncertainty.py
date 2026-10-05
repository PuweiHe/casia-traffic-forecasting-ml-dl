"""Fixed seed42 paired time-block diagnostics for already locked PEMS models.

This diagnostic reconstructs temporal error sums unavailable in aggregate score
files. It does not select models or replace the frozen primary test evaluation.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from training_safety import atomic_record
from locked_pems_evaluation import ROOT, REPO, digest
sys.path.insert(0, str(REPO))
from traffic_forecasting import multihorizon_mps as mps
from traffic_forecasting.multihorizon_data import load_development, MaskedWindows
from traffic_forecasting.multihorizon_study import predict


def bootstrap(blocks, replicates=5000, seed=911):
    """Resample paired contiguous blocks, preserving masks and horizon weights."""
    if len(blocks) < 2 or blocks.shape[1:] != (3, 12):
        raise ValueError('Expected at least two paired blocks with 12 horizons')
    rng = np.random.default_rng(seed)
    sampled = blocks[rng.integers(0, len(blocks), (replicates, len(blocks)))].sum(1)
    if np.any(sampled[:, 2] <= 0):
        raise ValueError('Empty resampled horizon')
    ref = (sampled[:, 0] / sampled[:, 2]).mean(1)
    candidate = (sampled[:, 1] / sampled[:, 2]).mean(1)
    return {'candidate_minus_reference_mph_ci95': np.quantile(candidate-ref,[.025,.975]).tolist(),
            'relative_reduction_percent_ci95': np.quantile(100*(1-candidate/ref),[.025,.975]).tolist()}


def main():
    os.umask(0o077)
    out = ROOT/'runs/final_pems_evaluation_v1'
    selection_path = out/'locked_selection.json'
    selection = json.loads(selection_path.read_text())
    declaration = {'version':'paired_daily_blocks_v1', 'selection_sha256':digest(selection_path),
                   'diagnostic_source_sha256':digest(Path(__file__)), 'seed':42,
                   'block_windows':288, 'resamples':5000, 'rng_seed':911,
                   'policy':'Consecutive non-overlapping 24h blocks of window starts; include final partial block; paired block resampling; fixed seed42 representative; conditional on this test period and these weights.'}
    with (out/'.uncertainty.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/'uncertainty_protocol.json'
        if path.exists() and json.loads(path.read_text()) != declaration:
            raise ValueError('Diagnostic protocol changed')
        if not path.exists():atomic_record(path,declaration)
        final=out/'paired_uncertainty.json'
        if final.exists():
            if json.loads(final.read_text())['protocol_sha256'] != digest(path):raise ValueError('Cache mismatch')
            print('Verified existing uncertainty result');return
        if not torch.backends.mps.is_available():raise RuntimeError('No MPS; no CPU fallback')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        models=[]
        for trial in ['stgcn_reference','stgcn_search_005']:
            row=next(r for r in selection['models'] if r['trial']==trial and r['seed']==42)
            checkpoint_path=ROOT/row['directory']/'selected.pt'
            if digest(checkpoint_path)!=row['hashes']['selected.pt']:raise ValueError('Changed weights')
            report=json.loads((out/f'{trial}_seed42.json').read_text())
            if report['selection_sha256']!=digest(selection_path):raise ValueError('Primary evaluation mismatch')
            models.append((row,torch.load(checkpoint_path,map_location='cpu',weights_only=True),report))
        values,graph,_,data=load_development(ROOT/'data','PEMS_BAY')
        frozen=ROOT/selection['models'][0]['directory'];frozen=frozen.parents[1]/'data_manifest.json'
        if json.loads(frozen.read_text())!=data:raise ValueError('Changed test data')
        windows=MaskedWindows(values,data['boundaries'][1],len(values),data['mean'],data['std'])
        loader=mps.MPSDataLoader(windows,batch_size=16,shuffle=False,num_workers=0)
        tensors=[]
        for _,checkpoint,_ in models:
            model=mps.build_model(checkpoint['trial'],graph).eval();model.load_state_dict(checkpoint['state_dict']);tensors.append(model)
        blocks=np.zeros(((len(windows)+287)//288,3,12),dtype=np.float64);offset=0
        with torch.inference_mode():
            for batch in loader:
                target=batch['y'].float().cpu().double();mask=batch['y_mask'].cpu()
                sums=[]
                for model in tensors:
                    predicted=predict(model,batch).float().cpu().double()
                    if not torch.isfinite(predicted).all():raise ValueError('Nonfinite prediction')
                    sums.append(((predicted-target).abs()*data['std']*mask).sum((2,3)).numpy())
                counts=mask.sum((2,3)).numpy()
                for i in range(len(counts)):
                    b=(offset+i)//288;blocks[b,0]+=sums[0][i];blocks[b,1]+=sums[1][i];blocks[b,2]+=counts[i]
                offset+=len(counts)
        totals=blocks.sum(0)
        mae=(totals[:2]/totals[2]).mean(1)
        for i,(_,_,report) in enumerate(models):
            if abs(mae[i]-report['primary_mae_mph'])>1e-8:raise ValueError('Diagnostic and frozen primary disagree')
        atomic_record(out/'paired_error_blocks.json',{'protocol_sha256':digest(path),'blocks':blocks.tolist(),'window_count':len(windows)})
        result={'protocol_sha256':digest(path),'seed':42,'block_count':len(blocks),'window_count':len(windows),
                'reference_mae_mph':float(mae[0]),'candidate_mae_mph':float(mae[1]),
                'candidate_minus_reference_mph':float(mae[1]-mae[0]),
                'relative_reduction_percent':float(100*(1-mae[1]/mae[0])),**bootstrap(blocks),
                'limits':['Conditional temporal-block uncertainty, not a guarantee for other dates or datasets.',
                          'One representative seed; report three-seed training variation separately.',
                          'Daily blocks approximate serial dependence; overlapping forecast windows are not independent samples.',
                          'Final partial block is resampled as observed; not a formal IID confidence guarantee.']}
        atomic_record(final,result);print(json.dumps(result))

if __name__=='__main__':main()

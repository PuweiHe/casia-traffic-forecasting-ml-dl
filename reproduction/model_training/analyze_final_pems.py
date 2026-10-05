"""Summarize immutable PEMS test evidence without running inference or selection."""
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    os.umask(0o077)
    root=ROOT/'runs/final_pems_evaluation_v1';selection=root/'locked_selection.json'
    lock=json.loads(selection.read_text());validation=json.loads((ROOT/'runs/analysis/pems_confirmation_validation/summary.json').read_text())
    rows=[];groups={}
    for trial in ['stgcn_reference','stgcn_search_005']:
        metrics=[]
        for seed in [17,42,73]:
            path=root/f'{trial}_seed{seed}.json';r=json.loads(path.read_text())
            if r['selection_sha256']!=sha(selection):raise ValueError('Wrong selection')
            original=next(x for x in validation['rows'] if x['trial']==trial and x['seed']==seed)
            rows.append({'configuration':trial,'seed':seed,'test_mae_mph':r['primary_mae_mph'],
                         'test_rmse_mph':r['test']['all']['overall']['rmse_mph'],
                         'parameters':original['parameters'],'timed_training_seconds':original['timed_training_seconds'],
                         'selected_epoch':original['selected_epoch'],'completed_epochs':original['completed_epochs'],
                         'test_report_path':str(path.relative_to(ROOT)),'test_report_sha256':sha(path),
                         'training_result_path':original['result_path'],'training_result_sha256':original['result_sha256']})
            metrics.append(r['test'])
        groups[trial]={}
        for group in metrics[0]:
            groups[trial][group]={}
            for horizon in ['overall','15','30','60']:
                points=[m[group]['overall'] if horizon=='overall' else m[group]['horizons'][horizon] for m in metrics]
                groups[trial][group][horizon]={k:statistics.mean(p[k] for p in points) if all(p[k] is not None for p in points) else None for k in ['mae_mph','rmse_mph']}
                groups[trial][group][horizon]['observations_per_seed']=points[0]['observations']
    summary={}
    for trial in groups:
        selected=[r for r in rows if r['configuration']==trial];maes=[r['test_mae_mph'] for r in selected]
        summary[trial]={'test_mae_mean_mph':statistics.mean(maes),'test_mae_sample_sd_mph':statistics.stdev(maes),
                        'parameters':selected[0]['parameters'],'total_timed_training_seconds_three_seeds':sum(r['timed_training_seconds'] for r in selected)}
    a=summary['stgcn_reference']['test_mae_mean_mph'];b=summary['stgcn_search_005']['test_mae_mean_mph']
    pairs=[{'seed':s,'candidate_minus_reference_mph':next(r['test_mae_mph'] for r in rows if r['seed']==s and r['configuration']=='stgcn_search_005')-next(r['test_mae_mph'] for r in rows if r['seed']==s and r['configuration']=='stgcn_reference')} for s in [17,42,73]]
    controls={}
    for name in ['persistence','moving_mean']:
        path=root/f'baseline_{name}.json';d=json.loads(path.read_text())
        if d['selection_sha256']!=sha(selection):raise ValueError('Wrong baseline selection')
        controls[name]={'test':d['test'],'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
    uncertainty=root/'paired_uncertainty.json'
    payload={'selection_sha256':sha(selection),'models':summary,'relative_mae_reduction_percent':100*(1-b/a),
             'paired_seed_differences':pairs,'groups':groups,'baselines':controls,'rows':rows,
             'paired_temporal_uncertainty':json.loads(uncertainty.read_text()) if uncertainty.exists() else {'status':'pending'},
             'limits':['Seed17 participated in validation screening; final test was reserved until all six models and selection were frozen.',
                       'Joint hyperparameter tuning, larger capacity and different batch size; not an isolated architectural innovation.',
                       'Three seeds quantify training variation, not uncertainty across cities or future years.',
                       'Existing-file provenance audit plus user attestation; deleted/external history cannot be technically ruled out.',
                       'Reconstructed measurements obtained in 2026, not recovered 2024 internship results.',
                       'No demonstrated deployment or congestion-reduction benefit.',
                       'Training seconds sum timed training/validation loops; exclude setup, checkpoint overhead and queue wait.']}
    out=ROOT/'runs/analysis/final_pems';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'trial_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Locked PEMS-BAY evaluation','',
           'Task: 12 past five-minute speed readings to predict the next 12, in mph. Chronological 70/10/20 partitions with partition-contained windows and training-only standardization. The exact validation selection, six checkpoint hashes, evaluation source and provenance ledger were frozen before scoring. No configuration was retuned using test results.','',
           '| Configuration | Test MAE mean ± sample SD (mph) | Parameters | Timed training, three seeds (h) |',
           '|---|---:|---:|---:|']
    for trial,d in summary.items():lines.append(f"| {trial} | {d['test_mae_mean_mph']:.6f} ± {d['test_mae_sample_sd_mph']:.6f} | {d['parameters']:,} | {d['total_timed_training_seconds_three_seeds']/3600:.3f} |")
    lines+=['',f"Joint tuning reduced mean test MAE by {payload['relative_mae_reduction_percent']:.3f}%; all three matched seeds improved. The tuned model has {summary['stgcn_search_005']['parameters']/summary['stgcn_reference']['parameters']:.2f} times as many parameters. This is an accuracy/cost trade-off, not a new STGCN architecture.",'',
            '| Configuration | 15-min MAE | 30-min MAE | 60-min MAE | Overall RMSE |','|---|---:|---:|---:|---:|']
    for trial,g in groups.items():lines.append(f"| {trial} | {g['all']['15']['mae_mph']:.6f} | {g['all']['30']['mae_mph']:.6f} | {g['all']['60']['mae_mph']:.6f} | {g['all']['overall']['rmse_mph']:.6f} |")
    lines+=['', '| Configuration | Slow (<30 mph) MAE | Moderate (30–55 mph) MAE | Free flow (≥55 mph) MAE | Missing history (>25%) MAE |','|---|---:|---:|---:|---:|']
    for trial,g in groups.items():
        cells=[g[n]['overall']['mae_mph'] for n in ['slow','moderate','free_flow','missing_history']]
        lines.append('| '+trial+' | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in cells)+' |')
    lines+=['', 'Baseline test MAE: '+', '.join(f"{name} {d['test']['all']['overall']['mae_mph']:.6f} mph" for name,d in controls.items())+'.','', '## Temporal uncertainty','',json.dumps(payload['paired_temporal_uncertainty'],indent=2),'', '## Attribution and limitations','']
    lines+=['- '+x for x in payload['limits']]
    lines+=['','Exact source files and hashes are in `trial_metrics.csv`; unrounded subgroup metrics and observation counts are in `summary.json`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary));print('Test reduction percent',payload['relative_mae_reduction_percent'])

if __name__=='__main__':main()

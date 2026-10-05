"""Summarize verified PEMS-BAY validation artifacts without loading test data."""
import hashlib
import json
import statistics
from pathlib import Path
from validate_training_artifacts import validate

ROOT = Path(__file__).resolve().parent

def main():
    screen = ROOT/'runs/local_pems_stgcn_mps_screen_v1'
    confirm = ROOT/'runs/local_pems_stgcn_mps_confirm_v1'
    final = ROOT/'runs/local_pems_stgcn_mps_seed73_v1'
    final_complete = (final/'PEMS_BAY/stgcn_search_005/73/result.json').exists()
    cohorts = [screen, confirm] + ([final] if final_complete else [])
    declarations = [json.loads((p/'frozen_protocol.json').read_text()) for p in cohorts]
    for extra in declarations[2:]:
        if extra['environment'] != declarations[0]['environment']:
            raise ValueError('Final replication environment mismatch')
        for name, digest in extra['code_sha256'].items():
            if name in declarations[0]['code_sha256'] and digest != declarations[0]['code_sha256'][name]:
                raise ValueError('Final shared source mismatch')
        for key in ['epochs','min_epochs','patience','threads','hardware','data']:
            if extra['protocol'][key] != declarations[0]['protocol'][key]:
                raise ValueError('Final numerical protocol mismatch: '+key)
        for trial in extra['protocol']['trials']:
            if trial not in declarations[0]['protocol']['trials']:
                raise ValueError('Final configuration mismatch')
    for key in ['environment']:
        if declarations[0][key] != declarations[1][key]:
            raise ValueError('Environment mismatch')
    shared_sources = set(declarations[0]['code_sha256']) & set(declarations[1]['code_sha256'])
    if any(declarations[0]['code_sha256'][k] != declarations[1]['code_sha256'][k] for k in shared_sources):
        raise ValueError('Shared numerical source mismatch')
    for key in ['epochs','min_epochs','patience','threads','hardware','data']:
        if declarations[0]['protocol'][key] != declarations[1]['protocol'][key]:
            raise ValueError('Protocol mismatch: '+key)
    before = {t['id']:t for t in declarations[0]['protocol']['trials']}
    for trial in declarations[1]['protocol']['trials']:
        if trial != before[trial['id']]:
            raise ValueError('Configuration changed during confirmation')
    manifests = [(p/'PEMS_BAY/data_manifest.json').read_bytes() for p in cohorts]
    if any(m != manifests[0] for m in manifests[1:]):
        raise ValueError('Data manifest mismatch')
    rows=[]
    for trial, seeds in [('stgcn_reference',[17,42,73]), ('stgcn_search_005',[17,42,73] if final_complete else [17,42])]:
        for seed in seeds:
            d=(screen if seed==17 else final if trial=='stgcn_search_005' and seed==73 else confirm)/'PEMS_BAY'/trial/str(seed)
            audit=validate(d)
            r=json.loads((d/'result.json').read_text())
            v=r['validation']['all']
            mae=statistics.mean(v['horizons'][str(h*5)]['mae_mph'] for h in range(1,13))
            if abs(mae-v['overall']['mae_mph'])>1e-10:
                raise ValueError('Pooled and equal-horizon MAE disagree')
            rows.append({'trial':trial,'seed':seed,'validation_mae_mph':mae,
                         'selected_epoch':r['selected_epoch'],'completed_epochs':r['completed_epochs'],
                         'parameters':r['parameters'],'timed_training_seconds':r['training_seconds'],
                         'result_path':str((d/'result.json').relative_to(ROOT)),
                         'result_sha256':audit['hashes']['result.json']})
    scores={(r['trial'],r['seed']):r['validation_mae_mph'] for r in rows}
    pairs=[{'seed':s,'candidate_minus_reference_mph':scores['stgcn_search_005',s]-scores['stgcn_reference',s],
            'relative_reduction_percent':100*(1-scores['stgcn_search_005',s]/scores['stgcn_reference',s])} for s in ([17,42,73] if final_complete else [17,42])]
    out=ROOT/'runs/analysis/pems_confirmation_validation';out.mkdir(parents=True,exist_ok=True)
    report={'status':'complete_three_seed_confirmation_validation_only' if final_complete else 'incomplete_three_seed_confirmation_validation_only','rows':rows,'paired_results':pairs,
            'data_manifest_sha256':hashlib.sha256(manifests[0]).hexdigest(),
            'limits':['Candidate seed73 verified complete' if final_complete else 'Candidate seed73 authorized and running under original8h budget',
                      'Seed 17 participated in 13-candidate selection; seeds42/73 are additional validation replications',
                      'PEMS-BAY provenance reviewed with disclosed audit scope; this script performs validation analysis only',
                      'Joint configuration search; larger parameter count, not a novel architecture gain',
                      'Timed training seconds exclude setup, checkpoint overhead and queue delay']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# PEMS-BAY bounded confirmation validation','',
           'The bounded three-task MPS confirmation completed successfully. All completed artifacts passed identity, best-weight and epoch-history verification. No new training or test evaluation was performed by this analysis.','',
           '| Configuration | Seed | Validation MAE (mph) | Selected epoch | Completed epochs | Parameters |',
           '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['trial']} | {r['seed']} | {r['validation_mae_mph']:.6f} | {r['selected_epoch']} | {r['completed_epochs']} | {r['parameters']} |" for r in rows]
    lines += ['', 'The candidate reduces validation MAE by '+', '.join(f"{p['relative_reduction_percent']:.3f}% for seed {p['seed']}" for p in pairs)+('. Seed17 was used for selection; seeds42/73 are additional replications. This is complete three-seed validation evidence, not independent-test improvement.' if final_complete else '. Seed17 was used for selection; seed42 is additional replication and seed73 is running. Confirmation remains incomplete.'), '', 'The candidate uses 637,516 parameters versus 35,940 for the reference. Configuration changes jointly affect graph order, temporal kernel, width, dropout, learning rate and batch size; these measurements do not isolate an individual change.', '', 'Candidate seed73 is independently bounded and cannot reset its deadline. Locked evaluation and final inference-cost assessment remain pending.', '', 'Exact source paths, result hashes, unrounded metrics and limits are in `summary.json`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_results':len(rows),'pairs':pairs}))

if __name__=='__main__':
    main()

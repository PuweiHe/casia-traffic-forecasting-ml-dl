"""Read-only completion validation for frozen STGCN/DCRNN/STTN studies."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from training_safety import atomic_record


def validate(directory):
    required=['result.json','status.json','selected.pt','progress.pt','epochs.json']
    for name in required:
        if not (directory/name).is_file():raise ValueError('Missing '+name)
    result=json.loads((directory/'result.json').read_text())
    status=json.loads((directory/'status.json').read_text())
    history=json.loads((directory/'epochs.json').read_text())
    selected=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    progress=torch.load(directory/'progress.pt',map_location='cpu',weights_only=True)
    identity=result['identity']
    if any(x['identity']!=identity for x in [status,selected,progress]):raise ValueError('Identity mismatch')
    if status['state']!='complete':raise ValueError('Result published without complete status')
    if result['selected_epoch']!=selected['epoch'] or result['selected_epoch']!=progress['best']['epoch']:
        raise ValueError('Selected epoch mismatch')
    if len(history)!=result['completed_epochs'] or len(progress['history'])!=len(history):
        raise ValueError('Epoch history mismatch')
    if selected['seed']!=result['seed'] or selected['trial']['id']!=result['trial']:
        raise ValueError('Trial/seed mismatch')
    if history!=progress['history'] or selected['validation']!=result['validation']:
        raise ValueError('Metric/history disagreement')
    if set(selected['state_dict'])!=set(progress['best']['state_dict']):raise ValueError('Best weight keys mismatch')
    for key,value in selected['state_dict'].items():
        if not torch.isfinite(value).all() or not torch.equal(value,progress['best']['state_dict'][key]):
            raise ValueError('Selected weight mismatch: '+key)
    manifest_path = next((p/'data_manifest.json' for p in directory.parents if (p/'data_manifest.json').exists()), None)
    declaration_path = next((p/'frozen_protocol.json' for p in directory.parents if (p/'frozen_protocol.json').exists()), None)
    if manifest_path and declaration_path:
        data=json.loads(manifest_path.read_text());declaration=json.loads(declaration_path.read_text())
        if selected.get('data_hashes')!=data['hashes'] or selected.get('normalization')!={'mean':data['mean'],'std':data['std']}:
            raise ValueError('Data hash/normalization mismatch')
        expected=[]
        for field in ['data','manifest']:
            payload={'declaration':declaration,field:data,'trial':selected['trial'],'seed':selected['seed']}
            expected.append(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
        if identity not in expected:raise ValueError('Frozen declaration/config identity mismatch')
    return {'trial':result['trial'],'seed':result['seed'],'identity':identity,'state':'verified_complete',
            'path':str(directory),'hashes':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in required}}


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.root.rglob('result.json')):
        try:rows.append(validate(path.parent))
        except Exception as e:rows.append({'path':str(path.parent),'state':'invalid_completion','error':str(e)})
    atomic_record(a.report,{'results':rows,'valid':bool(rows) and all(r['state']=='verified_complete' for r in rows),
                            'note':'No checkpoint, metric or completion metadata was modified.'})
    if any(r['state']=='invalid_completion' for r in rows):raise SystemExit(1)
    print('Verified',len(rows),'completed artifact sets')


if __name__=='__main__':main()

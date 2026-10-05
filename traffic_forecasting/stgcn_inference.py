"""Versioned PEMS-BAY STGCN serving artifacts and bounded JSON inference.

This is a validation-candidate artifact, not a final evaluated model promotion.
No training module or checkpoint is modified by packaging or serving.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from .multihorizon_study import build_model, predict

VERSION = 'pems_stgcn_inference_v1'
MAX_BATCH = 32


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def export_candidate(selected_path, data_dir, output_path):
    """Package trusted local weights with frozen graph, sensor order and scaler."""
    selected_path, output_path = Path(selected_path), Path(output_path)
    if output_path.exists():
        raise ValueError('Artifact already exists; refusing overwrite')
    checkpoint = torch.load(selected_path, map_location='cpu', weights_only=True)
    trial = checkpoint['trial']
    if trial['model'] != 'STGCN' or not trial['config'].get('direct_multi_step'):
        raise ValueError('This artifact supports direct-output STGCN only')
    manifest_path = next((p/'data_manifest.json' for p in selected_path.parents
                          if (p/'data_manifest.json').is_file()), None)
    declaration_path = next((p/'frozen_protocol.json' for p in selected_path.parents
                             if (p/'frozen_protocol.json').is_file()), None)
    if manifest_path is None or declaration_path is None:
        raise ValueError('Frozen manifest/declaration missing')
    manifest = json.loads(manifest_path.read_text())
    if manifest['dataset'] != 'PEMS_BAY' or manifest['unit'] != 'mph':
        raise ValueError('PEMS-BAY mph manifest required')
    expected_norm = {'mean':manifest['mean'], 'std':manifest['std']}
    if checkpoint['data_hashes'] != manifest['hashes'] or checkpoint['normalization'] != expected_norm:
        raise ValueError('Checkpoint data/scaler provenance mismatch')
    declaration = json.loads(declaration_path.read_text())
    identity_payload = {'declaration':declaration, 'data':manifest, 'trial':trial, 'seed':checkpoint['seed']}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    if checkpoint['identity'] != identity:
        raise ValueError('Checkpoint frozen identity mismatch')
    repo = Path(__file__).resolve().parents[1]
    for name, expected in declaration['code_sha256'].items():
        if file_sha(repo/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    data_dir = Path(data_dir)
    for name in ['adj.npy','sensor_ids.npy']:
        if file_sha(data_dir/name) != manifest['hashes'][name]:
            raise ValueError('Graph/sensor provenance mismatch')
    graph = np.load(data_dir/'adj.npy', allow_pickle=False).astype(np.float32)
    sensors = [str(s) for s in np.load(data_dir/'sensor_ids.npy', allow_pickle=False).tolist()]
    if graph.shape != (len(sensors),len(sensors)) or len(set(sensors)) != len(sensors):
        raise ValueError('Invalid graph or duplicate sensor IDs')
    artifact = {'version':VERSION, 'stage':'validation_candidate_not_final_test',
                'identity':identity, 'trial':trial, 'seed':checkpoint['seed'],
                'normalization':expected_norm, 'unit':'mph', 'history':12, 'horizon':12,
                'step_seconds':300, 'sensor_ids':sensors, 'graph':torch.from_numpy(graph.copy()),
                'graph_array_sha256':array_sha(graph), 'state_dict':checkpoint['state_dict'],
                'source_checkpoint_sha256':file_sha(selected_path),
                'frozen_declaration_sha256':file_sha(declaration_path),
                'data_manifest_sha256':file_sha(manifest_path), 'data_hashes':manifest['hashes'],
                'training_source_sha256':declaration['code_sha256'],
                'packaging_source_sha256':file_sha(__file__),
                'training_environment':declaration['environment'],
                'selected_epoch':checkpoint['epoch']}
    # Validate reload structure and model keys before publishing the artifact.
    Predictor(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(artifact, temporary)
        temporary.chmod(0o600)
        os.link(temporary, output_path)  # Atomic, exclusive publication; no overwrite race.
    finally:
        if temporary.exists():temporary.unlink()
    return {'artifact_sha256':file_sha(output_path), 'identity':identity, 'stage':artifact['stage']}


class Predictor:
    """Load once and reuse; inputs are raw mph plus explicit JSON null missingness."""
    def __init__(self, artifact, device='cpu'):
        if isinstance(artifact, (str, Path)):
            artifact = torch.load(artifact, map_location='cpu', weights_only=True)
        if artifact['version'] != VERSION or artifact['unit'] != 'mph':
            raise ValueError('Unsupported artifact format/unit')
        if (artifact['history'],artifact['horizon'],artifact['step_seconds']) != (12,12,300):
            raise ValueError('Unsupported temporal contract')
        if artifact['packaging_source_sha256'] != file_sha(__file__):
            raise ValueError('Packaging code changed; create a new artifact version')
        graph = artifact['graph'].numpy()
        if array_sha(graph) != artifact['graph_array_sha256'] or not np.isfinite(graph).all() or (graph<0).any():
            raise ValueError('Invalid graph fingerprint/values')
        sensors = artifact['sensor_ids']
        if not sensors or len(set(sensors)) != len(sensors) or graph.shape != (len(sensors),len(sensors)):
            raise ValueError('Sensor order/graph mismatch')
        mean, std = artifact['normalization']['mean'], artifact['normalization']['std']
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('Invalid normalization')
        if device not in ['cpu','mps','cuda']:
            raise ValueError('Explicit cpu/mps/cuda device required')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS unavailable; no fallback')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; no fallback')
        repo=Path(__file__).resolve().parents[1]
        for name, expected in artifact['training_source_sha256'].items():
            if file_sha(repo/name) != expected:
                raise ValueError('Model/preprocessing source differs from artifact')
        self.artifact, self.device = artifact, torch.device(device)
        self.model = build_model(artifact['trial'],graph).to(self.device).eval()
        self.model.load_state_dict(artifact['state_dict'], strict=True)
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError('Nonfinite model weights')

    def prepare(self, payload):
        if not isinstance(payload,dict) or set(payload) != {'sensor_ids','timestamps','speed_mph'}:
            raise ValueError('Request needs only sensor_ids, timestamps, speed_mph')
        if payload['sensor_ids'] != self.artifact['sensor_ids']:
            raise ValueError('Sensor IDs must match artifact order exactly')
        raw=payload['speed_mph'];n=len(self.artifact['sensor_ids'])
        if not isinstance(raw,list) or not 1 <= len(raw) <= MAX_BATCH:
            raise ValueError('Batch must contain 1 to 32 windows')
        # Validate nesting before NumPy conversion; reject booleans and strings.
        if any(not isinstance(window,list) or len(window)!=12 for window in raw):
            raise ValueError('Each window requires 12 history steps')
        for window in raw:
            for step in window:
                if not isinstance(step,list) or len(step)!=n:
                    raise ValueError('Incorrect sensor dimension')
                for feature in step:
                    if not isinstance(feature,list) or len(feature)!=1:
                        raise ValueError('Exactly one speed feature required')
                    value=feature[0]
                    if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
                        raise ValueError('Speeds must be finite JSON numbers or null')
        x=np.asarray(raw,dtype=np.float32)
        # JSON null -> NaN -> missing. Nonpositive speeds match the frozen training mask.
        finite_input=np.asarray(raw,dtype=object)
        if np.any(~np.isfinite(x) & (finite_input != None)):
            raise ValueError('Speed overflows float32')
        observed=np.isfinite(x) & (x>0)
        mean,std=(self.artifact['normalization'][k] for k in ['mean','std'])
        normalized=np.where(observed,(x-mean)/std,0).astype(np.float32)
        if not np.isfinite(normalized).all():raise ValueError('Normalization overflow')
        timestamps=payload['timestamps']
        if not isinstance(timestamps,list) or len(timestamps)!=len(raw):
            raise ValueError('One timestamp history per batch window required')
        future=[]
        for history in timestamps:
            if not isinstance(history,list) or len(history)!=12 or any(not isinstance(t,str) for t in history):
                raise ValueError('Twelve ISO timestamps per window required')
            try:times=[dt.datetime.fromisoformat(t.replace('Z','+00:00')) for t in history]
            except ValueError as exc:raise ValueError('Invalid ISO timestamp') from exc
            if any(t.utcoffset() is None for t in times):raise ValueError('Timezone required')
            times=[t.astimezone(dt.timezone.utc) for t in times]
            if any(t.second or t.microsecond or t.minute%5 for t in times):raise ValueError('Five-minute boundaries required')
            if any((b-a).total_seconds()!=300 for a,b in zip(times,times[1:])):
                raise ValueError('Consecutive five-minute history required')
            future.append([(times[-1]+dt.timedelta(minutes=5*i)).isoformat() for i in range(1,13)])
        batch={'X':torch.from_numpy(normalized).to(self.device),
               'X_mask':torch.from_numpy(observed).to(self.device)}
        return batch,future,float((~observed).mean())

    @torch.inference_mode()
    def predict(self, payload):
        batch,timestamps,missing = self.prepare(payload)
        values = predict(self.model,batch).cpu().numpy()
        if values.shape != batch['X'].shape or not np.isfinite(values).all():
            raise RuntimeError('Invalid prediction shape or values')
        norm=self.artifact['normalization'];values=values*norm['std']+norm['mean']
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite mph output')
        return {'version':VERSION,'artifact_identity':self.artifact['identity'],
                'stage':self.artifact['stage'],'sensor_ids':self.artifact['sensor_ids'],
                'timestamps':timestamps,'speed_mph':values.tolist(),'missing_history_fraction':missing}



def make_handler(predictor):
    """Bounded loopback demo service; model is loaded once before accepting requests."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log request contents or sensor histories.

        def reply(self, status, body):
            encoded=json.dumps(body,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)));self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path=='/health':self.reply(200,{'status':'ready','version':VERSION})
            else:self.reply(404,{'error':'Unknown route'})

        def do_POST(self):
            if self.path!='/predict':
                self.reply(404,{'error':'Unknown route'});self.close_connection=True;return
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:length=0
            if not 0<length<=8*1024*1024:
                self.reply(413,{'error':'Invalid or oversized body'});self.close_connection=True;return
            if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                self.reply(415,{'error':'JSON body required'});self.close_connection=True;return
            self.connection.settimeout(15)
            try:
                def reject_constant(_value):raise ValueError('Nonstandard JSON numeric constant')
                payload=json.loads(self.rfile.read(length),parse_constant=reject_constant)
                result=predictor.predict(payload)
            except (ValueError,TypeError,UnicodeDecodeError) as exc:
                self.reply(400,{'error':str(exc)});return
            except TimeoutError:
                self.close_connection=True;return
            except Exception:
                self.reply(500,{'error':'Inference failed'});return
            self.reply(200,result)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    export=sub.add_parser('export');export.add_argument('--selected',type=Path,required=True)
    export.add_argument('--data-dir',type=Path,required=True);export.add_argument('--output',type=Path,required=True)
    infer=sub.add_parser('predict');infer.add_argument('--artifact',type=Path,required=True)
    infer.add_argument('--request',type=Path,required=True);infer.add_argument('--output',type=Path,required=True)
    infer.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    serve=sub.add_parser('serve');serve.add_argument('--artifact',type=Path,required=True)
    serve.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');serve.add_argument('--port',type=int,default=8080)
    a=p.parse_args();torch.set_num_threads(4)
    if a.mode=='export':print(json.dumps(export_candidate(a.selected,a.data_dir,a.output)))
    elif a.mode=='serve':
        predictor=Predictor(a.artifact,a.device)
        with HTTPServer(('127.0.0.1',a.port),make_handler(predictor)) as server:
            try:server.serve_forever()
            except KeyboardInterrupt:pass
    else:
        if a.output.exists():raise ValueError('Refusing to overwrite output')
        result=Predictor(a.artifact,a.device).predict(json.loads(a.request.read_text()))
        a.output.write_text(json.dumps(result,allow_nan=False)+'\n')

if __name__=='__main__':main()

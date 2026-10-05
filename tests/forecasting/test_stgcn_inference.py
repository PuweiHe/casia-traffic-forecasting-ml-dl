import copy
import datetime as dt
import hashlib
import http.client
import json
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch

import numpy as np
import torch

from traffic_forecasting import stgcn_inference as serving
from traffic_forecasting.multihorizon_data import MaskedWindows
from traffic_forecasting.multihorizon_study import build_model, predict


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.graph=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.float32)
        cls.trial={'id':'synthetic','model':'STGCN','config':{'Ks':2,'Kt':3,'blocks':[[1,2,4],[4,2,4]],
                   'dropout':.1,'graph_conv_type':'chebconv','direct_multi_step':True},'batch_size':2,'learning_rate':.001}
        cls.model=build_model(cls.trial,cls.graph).eval()
        cls.artifact={'version':serving.VERSION,'unit':'mph','history':12,'horizon':12,'step_seconds':300,
                      'sensor_ids':['a','b','c'],'graph':torch.from_numpy(cls.graph),
                      'graph_array_sha256':serving.array_sha(cls.graph),'normalization':{'mean':10.,'std':2.},
                      'packaging_source_sha256':serving.file_sha(serving.__file__),'training_source_sha256':{},
                      'trial':cls.trial,'identity':'synthetic','stage':'validation_candidate_not_final_test',
                      'state_dict':cls.model.state_dict()}
        cls.engine=serving.Predictor(cls.artifact)

    def payload(self):
        start=dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)
        raw=np.arange(36,dtype=float).reshape(1,12,3,1)+1
        raw[0,-1,0,0]=np.nan;raw[0,-2,1,0]=0;raw[0,-3,2,0]=10
        values=raw.tolist();values[0][-1][0][0]=None
        return {'sensor_ids':['a','b','c'],'timestamps':[[(start+dt.timedelta(minutes=5*i)).isoformat() for i in range(12)]],
                'speed_mph':values},raw

    def test_preprocessing_and_prediction_match_training_path(self):
        payload,raw=self.payload();values=np.concatenate([raw[0,:,:,0],np.ones((12,3))]).astype(np.float32)
        reference=MaskedWindows(values,0,24,10.,2.)[0]
        inputs,_,_=self.engine.prepare(payload)
        torch.testing.assert_close(inputs['X'],reference['X'][None],rtol=0,atol=0)
        self.assertTrue(torch.equal(inputs['X_mask'],reference['X_mask'][None]))
        expected=(predict(self.model,inputs).detach().numpy()*2+10).tolist()
        result=self.engine.predict(payload)
        np.testing.assert_array_equal(result['speed_mph'],expected)
        self.assertEqual(result['timestamps'][0][0],'2026-01-01T01:00:00+00:00')
        self.assertFalse(self.engine.model.training)
        self.assertEqual(self.engine.predict(payload)['speed_mph'],result['speed_mph'])

    def test_schema_sensor_order_and_timestamps_rejected(self):
        original,_=self.payload()
        cases=[]
        p=copy.deepcopy(original);p['sensor_ids'].reverse();cases.append(p)
        p=copy.deepcopy(original);p['y']=[];cases.append(p)
        p=copy.deepcopy(original);p['speed_mph'][0][0][0][0]=True;cases.append(p)
        p=copy.deepcopy(original);p['speed_mph'][0][0][0][0]=float('inf');cases.append(p)
        p=copy.deepcopy(original);p['timestamps'][0][1]=p['timestamps'][0][0];cases.append(p)
        p=copy.deepcopy(original);p['timestamps'][0]=[t[:-6] for t in p['timestamps'][0]];cases.append(p)
        p=copy.deepcopy(original);p['speed_mph'][0].pop();cases.append(p)
        p=copy.deepcopy(original);p['speed_mph']=[];cases.append(p)
        for p in cases:
            with self.assertRaises(ValueError):self.engine.predict(p)

    def test_all_missing_and_unavailable_device(self):
        payload,_=self.payload();payload['speed_mph']=[[[[None] for _ in range(3)] for _ in range(12)]]
        inputs,_,fraction=self.engine.prepare(payload)
        self.assertEqual(fraction,1.)
        self.assertEqual(inputs['X'].count_nonzero().item(),0)
        self.assertFalse(inputs['X_mask'].any())
        self.assertTrue(np.isfinite(self.engine.predict(payload)['speed_mph']).all())
        with patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'no fallback'):serving.Predictor(self.artifact,'mps')

    def test_tampered_graph_is_rejected(self):
        artifact=copy.deepcopy(self.artifact);artifact['graph'][0,1]=.25
        with self.assertRaisesRegex(ValueError,'graph'):serving.Predictor(artifact)

    def test_http_contract_and_invalid_body(self):
        with HTTPServer(('127.0.0.1',0),serving.make_handler(self.engine)) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                def request(method,path,body=None,headers=None):
                    connection=http.client.HTTPConnection(*server.server_address,timeout=5)
                    connection.request(method,path,body,headers or {})
                    response=connection.getresponse();value=json.loads(response.read());status=response.status
                    connection.close();return status,value
                self.assertEqual(request('GET','/health')[0],200)
                payload,_=self.payload()
                status,body=request('POST','/predict',json.dumps(payload),{'Content-Type':'application/json'})
                self.assertEqual(status,200);self.assertEqual(len(body['speed_mph'][0]),12)
                self.assertEqual(request('POST','/predict','{}',{'Content-Type':'application/json'})[0],400)
                self.assertEqual(request('POST','/predict','{}',{'Content-Type':'text/plain'})[0],415)
                self.assertEqual(request('GET','/other')[0],404)
            finally:server.shutdown();thread.join(timeout=5)

if __name__=='__main__':unittest.main()

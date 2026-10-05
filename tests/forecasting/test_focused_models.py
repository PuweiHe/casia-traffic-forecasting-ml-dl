import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.multihorizon_data import MaskedWindows
from traffic_forecasting.focused_models import SpeedSTAEformer
from traffic_forecasting.cuda_search import horizon_objective

class FocusedModelTests(unittest.TestCase):
    def test_interrupted_resume_restores_dropout_and_optimizer(self):
        torch.set_num_threads(1)
        config = dict(input_embedding_dim=8, adaptive_embedding_dim=8,
                      num_heads=4, num_layers=1, feed_forward_dim=16,
                      dropout=.2, embedding_mode='adaptive')
        trial = dict(id='resume_test', model='STAEformer', config=config,
                     learning_rate=.001, weight_decay=.0001, batch_size=3)
        values = np.random.default_rng(4).uniform(1, 20, (30, 3)).astype(np.float32)
        windows = MaskedWindows(values, 0, 30, 10., 2.)
        train = DataLoader(windows, batch_size=3, shuffle=True)
        val = DataLoader(windows, batch_size=3)
        manifest = dict(std=2., mean=10., hashes={}, dataset='synthetic')
        protocol = dict(epochs=3, min_epochs=3, patience=4)
        graph = np.eye(3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                study, 'build_model', side_effect=lambda t, g: SpeedSTAEformer(t['config'], len(g))):
            full, resumed = Path(tmp)/'full', Path(tmp)/'resumed'
            study.train_trial(trial, 42, graph, train, val, manifest, protocol, full, 'unit')
            real_score, calls = study.score, [0]
            def interrupt(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('simulated interruption')
                return real_score(*args, **kwargs)
            with patch.object(study, 'score', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    study.train_trial(trial, 42, graph, train, val, manifest, protocol, resumed, 'unit')
            study.train_trial(trial, 42, graph, train, val, manifest, protocol, resumed, 'unit')
            a = torch.load(full/'progress.pt', weights_only=True)
            b = torch.load(resumed/'progress.pt', weights_only=True)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['best']['epoch'], b['best']['epoch'])
            self.assertEqual(a['scheduler'], b['scheduler'])
    def test_all_embeddings_and_reload(self):
        torch.set_num_threads(1)
        for mode in ('adaptive','zero','spatial'):
            c=dict(input_embedding_dim=8,adaptive_embedding_dim=8,num_heads=4,num_layers=1,feed_forward_dim=16,dropout=0.,embedding_mode=mode)
            m=SpeedSTAEformer(c,5)
            batch={'X':torch.randn(2,12,5,1),'y':torch.randn(2,12,5,1)}
            y=m(batch)
            self.assertEqual(tuple(y.shape),(2,12,5,1))
            y.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            m.eval(); other=SpeedSTAEformer(c,5); other.load_state_dict(m.state_dict()); other.eval()
            self.assertTrue(torch.equal(m(batch),other(batch)))
            self.assertTrue(torch.equal(m(batch),m(dict(batch,y=torch.zeros_like(batch['y'])))))
    def test_invalid_heads(self):
        with self.assertRaises(ValueError):
            SpeedSTAEformer(dict(input_embedding_dim=7,adaptive_embedding_dim=8,num_heads=4),5)
    def test_objective_is_not_pooled(self):
        metrics={'all':{'overall':{'mae_mph':9.},'horizons':{str(h*5):{'mae_mph':float(h)} for h in range(1,13)}}}
        result=horizon_objective(metrics)
        self.assertEqual(result['all']['overall']['mae_mph'],6.5)
        self.assertEqual(result['all']['overall']['pooled_mae_mph'],9.)
        self.assertEqual(metrics['all']['overall']['mae_mph'],9.)
if __name__=='__main__':unittest.main()

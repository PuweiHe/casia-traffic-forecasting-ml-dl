"""Focused checks for the independent local linear forecasting baselines."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from traffic_forecasting.local_linear import LinearForecast, make_batch, score, train_one


class LocalLinearTests(unittest.TestCase):
    def test_missing_history_and_future_labels_do_not_change_inference(self):
        for kind in ('linear', 'dlinear'):
            with self.subTest(kind=kind):
                model = LinearForecast(kind)
                x = torch.tensor([[[[1.], [0.]]] * 12])
                mask = torch.tensor([[[[True], [False]]] * 12])
                predicted = model(x, mask)
                self.assertEqual(predicted.shape, (1, 12, 2, 1))
                self.assertTrue(torch.equal(predicted[0, :, 0, 0], torch.ones(12)))
                self.assertTrue(torch.equal(predicted[0, :, 1, 0], torch.zeros(12)))
                x[0, 0, 1, 0] = 999
                self.assertTrue(torch.equal(model(x, mask), predicted))
                model(x, mask).sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))


    def test_partition_windows_never_cross_boundary(self):
        values = np.arange(90, dtype=np.float32).reshape(45, 2) + 1
        x, observed, y, valid = make_batch(values, np.array([0, 1]), 0., 1., torch.device('cpu'))
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, (2, 12, 2, 1))
        self.assertTrue(observed.all() and valid.all())
        self.assertEqual(x[1, 0, 0, 0], values[1, 0])
        self.assertEqual(y[1, -1, 0, 0], values[24, 0])
        metrics = score(None, values, (0, 25), 0., 1., 2, torch.device('cpu'), 'persistence')
        self.assertEqual(metrics['observations'], 48)
        with self.assertRaisesRegex(ValueError, 'no observed'):
            score(None, np.zeros_like(values), (0, 25), 0., 1., 2, torch.device('cpu'), 'persistence')


    def test_checkpoint_resume_keeps_completed_result(self):
        values = (np.arange(72, dtype=np.float32)[:, None] + 10).repeat(2, axis=1)
        manifest = {'dataset': 'METR_LA', 'hashes': {'synthetic': 'test'},
                    'boundaries': [48, 72], 'mean': 40., 'std': 20.}
        protocol = {'epochs': 2, 'min_epochs': 1, 'patience': 1,
                    'batch_size': 4, 'learning_rate': .001, 'weight_decay': 0.}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'new_study'
            first = train_one(root, 'METR_LA', 'linear', 17, values, manifest,
                              protocol, torch.device('cpu'))
            self.assertEqual(first['epochs'], 2)
            self.assertTrue((root / 'METR_LA/linear/17/selected.pt').exists())
            second = train_one(root, 'METR_LA', 'linear', 17, values, manifest,
                               protocol, torch.device('cpu'))
            self.assertEqual(second, first)
            self.assertEqual(json.loads((root / 'METR_LA/linear/17/result.json').read_text()), first)
            (root / 'METR_LA/linear/17/result.json').unlink()
            (root / 'METR_LA/linear/17/selected.pt').unlink()
            resumed = train_one(root, 'METR_LA', 'linear', 17, values, manifest,
                                protocol, torch.device('cpu'))
            self.assertEqual(resumed, first)
            self.assertTrue((root / 'METR_LA/linear/17/selected.pt').exists())


if __name__ == '__main__':
    unittest.main()

"""Boundary and resume checks for the independent nonlinear temporal baseline."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from traffic_forecasting.local_mlp import TemporalMLP, train


class TemporalMLPTests(unittest.TestCase):
    def test_mask_controls_missing_history_and_gradients(self):
        model = TemporalMLP(8)
        x = torch.ones(2, 12, 3, 1)
        observed = torch.ones_like(x, dtype=torch.bool)
        observed[:, :, 1] = False
        first = model(x, observed)
        x[:, :, 1] = 5000
        self.assertTrue(torch.equal(first, model(x, observed)))
        self.assertEqual(first.shape, (2, 12, 3, 1))
        first.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_complete_checkpoint_can_be_resumed(self):
        values = (np.arange(72, dtype=np.float32)[:, None] + 10).repeat(2, axis=1)
        manifest = {'dataset': 'METR_LA', 'hashes': {'synthetic': 'test'},
                    'boundaries': [48, 72], 'mean': 40., 'std': 20.}
        protocol = {'width': 8, 'epochs': 2, 'min_epochs': 1, 'patience': 1,
                    'batch_size': 4, 'learning_rate': .001, 'weight_decay': 0.}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = train(root, 'METR_LA', values, manifest, protocol, torch.device('cpu'), 17)
            self.assertEqual(first['epochs'], 2)
            out = root / 'METR_LA/temporal_mlp/17'
            (out / 'result.json').unlink()
            (out / 'selected.pt').unlink()
            second = train(root, 'METR_LA', values, manifest, protocol, torch.device('cpu'), 17)
            self.assertEqual(second, first)
            self.assertTrue((out / 'selected.pt').exists())
            self.assertEqual(json.loads((out / 'result.json').read_text()), first)


if __name__ == '__main__':
    unittest.main()

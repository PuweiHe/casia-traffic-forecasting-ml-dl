import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_forecasting.cached_windows import CachedMaskedWindows
from traffic_forecasting.multihorizon_data import MaskedWindows


class CachedWindowsTests(unittest.TestCase):
    def test_exact_samples_batches_order_and_rng(self):
        values = np.random.default_rng(8).uniform(1, 70, (100, 5)).astype(np.float32)
        values[12, 0] = np.nan
        values[13, 1] = 0
        values[14, 2] = -1
        values[15, 3] = np.inf
        for dtype in (np.float32, np.float64):
            args = (values.astype(dtype), 7, 90, 31.7, 12.3)
            reference = MaskedWindows(*args, stride=2)
            cached = CachedMaskedWindows(*args, stride=2)
            for i in range(len(reference)):
                for key, expected in reference[i].items():
                    actual = cached[i][key]
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual, expected)
            for shuffle in (False, True):
                torch.manual_seed(17)
                expected = list(DataLoader(reference, batch_size=8, shuffle=shuffle))
                expected_rng = torch.get_rng_state()
                torch.manual_seed(17)
                actual = list(DataLoader(cached, batch_size=8, shuffle=shuffle))
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertEqual(len(expected), len(actual))
                for a, b in zip(actual, expected):
                    for key in a:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_returned_samples_cannot_mutate_cache(self):
        values = np.ones((50, 3), dtype=np.float32)
        cached = CachedMaskedWindows(values, 0, 50, 0, 1)
        cached[0]['X'].zero_()
        cached.__getitems__([0, 1])[0]['X'].zero_()
        self.assertTrue(cached[0]['X'].eq(1).all())


if __name__ == '__main__':
    unittest.main()

"""Precompute partition-local normalization without changing window semantics."""
import numpy as np
import torch

from traffic_forecasting.multihorizon_data import MaskedWindows


class CachedMaskedWindows(MaskedWindows):
    """Use the reference arithmetic once per partition, with independent samples."""

    def __init__(self, values, begin, end, mean, std, history=12, horizon=12, stride=1):
        super().__init__(values, begin, end, mean, std, history, horizon, stride)
        raw = values[begin:end]
        mask = np.isfinite(raw) & (raw > 0)
        normalized = np.where(mask, (raw - mean) / std, 0).astype(np.float32)
        self.normalized = torch.from_numpy(normalized[..., None])
        self.observed = torch.from_numpy(mask[..., None])
        self.begin = begin

    def __getitem__(self, index):
        start = int(self.starts[index])
        offset = start - self.begin
        middle = offset + self.history
        stop = middle + self.horizon
        return {'X': self.normalized[offset:middle].clone(),
                'X_mask': self.observed[offset:middle].clone(),
                'y': self.normalized[middle:stop].clone(),
                'y_mask': self.observed[middle:stop].clone(), 'start': start}

    def __getitems__(self, indices):
        # Gather a batch at once; default_collate preserves the reference schema.
        starts = self.starts[np.asarray(indices)]
        offsets = torch.as_tensor(starts - self.begin)[:, None]
        positions = offsets + torch.arange(self.history + self.horizon)[None, :]
        normalized = self.normalized[positions]
        observed = self.observed[positions]
        return [{'X': normalized[i, :self.history],
                 'X_mask': observed[i, :self.history],
                 'y': normalized[i, self.history:],
                 'y_mask': observed[i, self.history:], 'start': int(start)}
                for i, start in enumerate(starts)]

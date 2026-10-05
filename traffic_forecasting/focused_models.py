"""Adapters for new focused protocols; historical model code stays immutable."""
import copy
import torch
from traffic_forecasting.metr_la_study import model_class

STAEformer = model_class("STAEformer")


class SpeedSTAEformer(torch.nn.Module):
    """Speed-only STAEformer with explicit embedding ablations."""
    def __init__(self, config, num_nodes):
        super().__init__()
        c = copy.deepcopy(config)
        c.update(input_window=12, output_window=12, input_dim=1, output_dim=1,
                 add_time_in_day=False, add_day_in_week=False,
                 tod_embedding_dim=0, dow_embedding_dim=0)
        mode = c.pop('embedding_mode', 'adaptive')
        if mode not in ('adaptive', 'zero', 'spatial'):
            raise ValueError('Unknown embedding ablation')
        width = c.get('adaptive_embedding_dim', 80)
        if (c.get('input_embedding_dim', 24) + width) % c.get('num_heads', 4):
            raise ValueError('Embedding width must be divisible by attention heads')
        c['spatial_embedding_dim'] = 0
        self.network = STAEformer(c, {'num_nodes': num_nodes})
        if mode == 'zero':
            self.network.adaptive_embedding.requires_grad_(False)
            with torch.no_grad():
                self.network.adaptive_embedding.zero_()
        elif mode == 'spatial':
            # Keep attention width fixed; fewer trainable parameters are disclosed.
            self.network.adaptive_embedding = torch.nn.Parameter(torch.empty(1, num_nodes, width))
            torch.nn.init.xavier_uniform_(self.network.adaptive_embedding)
        self.embedding_mode = mode

    def forward(self, batch):
        x = batch['X']
        if x.ndim != 4 or x.shape[1] != 12 or x.shape[-1] != 1:
            raise ValueError('Expected [batch,12,nodes,1] normalized speeds')
        # Spatial-only embedding must broadcast along the input-time axis.
        if self.embedding_mode == 'spatial':
            # Broadcast the shared spatial embedding without mutating parameters.
            return self._spatial_forward(x)
        return self.network(x)

    def _spatial_forward(self, x):
        n = self.network
        features = torch.cat((n.input_proj(x), n.adaptive_embedding.unsqueeze(0).expand(x.shape[0], 12, -1, -1)), -1)
        for layer in n.attn_layers_t:
            features = layer(features, dim=1)
        for layer in n.attn_layers_s:
            features = layer(features, dim=2)
        if not n.use_mixed_proj:
            raise ValueError('Spatial ablation requires mixed projection')
        return n.output_proj(features.transpose(1, 2).reshape(x.shape[0], n.num_nodes, -1)).view(x.shape[0], n.num_nodes, 12, 1).transpose(1, 2)

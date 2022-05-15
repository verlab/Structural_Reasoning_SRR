"""
    ScaleConvLayer.
"""


import dgl.function as fn
import torch
import torch.nn as nn
from tools.utils_model import get_activation_func, get_aggregation_func


class SocialScaleConv3x(nn.Module):

    def __init__(
        self,
        scales,
        size,
        activation,
        aggregation,
        normalization,
        dropout=0.0
    ):

        super(SocialScaleConv3x, self).__init__()

        # Input parameters
        self._scales = scales
        self._size = size
        self._activation = activation
        self._aggregation = aggregation
        self._normalization = normalization
        self._dropout = dropout

        # Callable Functions
        self._func_activation = get_activation_func(self._activation)
        self._func_aggregation = get_aggregation_func(self._aggregation)

        # Layer parameters
        self._layer_scales = nn.ModuleDict()
        self._layer_dropout = nn.Dropout(self._dropout)

        for scale in self._scales:
            self._layer_scales[scale] = nn.Linear(
                self._size, self._size, bias=True)

        if (self._normalization == 'batch'):
            self._layer_norm = nn.BatchNorm1d(self._size * 3)

        elif (self._normalization == 'layer'):
            self._layer_norm = nn.LayerNorm(
                self._size * 3, eps=1e-6, elementwise_affine=True)

        # Parameters initialization
        self.init_params(self._activation)

    @torch.no_grad()
    def init_params(self, activation):
        """Initialize layer parameters."""

        gain = nn.init.calculate_gain(activation)
        for scale in self._scales:
            nn.init.xavier_uniform_(
                self._layer_scales[scale].weight, gain=gain)
            nn.init.zeros_(self._layer_scales[scale].bias)

    def message_func(self, nodes):
        """Custom message function."""

        return {'features': self._func_aggregation(nodes.mailbox['message'], dim=1)}

    def forward(self, graph, features):
        """Forward step."""

        outputs = []

        # Get relation subgraph
        for scale in self._scales:

            # Check features
            assert scale in features, \
                '>> [ERROR] Inputs missing %s scale features' % scale

            # Check graph
            assert scale in graph.etypes, \
                '>> [ERROR] Graph missing %s scale edges' % scale

            # Apply dropout
            scale_features = self._layer_dropout(features[scale])

            # Apply weights
            scale_features = self._layer_scales[scale](scale_features)

            # Get subgraph
            subgraph = graph.edge_type_subgraph([scale])

            # Message passing
            with subgraph.local_scope():

                subgraph.nodes[scale].data['features'] = scale_features
                subgraph.update_all(fn.copy_src(
                    'features', 'message'), self.message_func)

                outputs.append(subgraph.nodes['relation'].data['features'])

        # Generate relation node features
        result = torch.cat(outputs, 1)

        # Apply activation
        result = self._func_activation(result)

        # Apply normalization
        if self._normalization:
            result = self._layer_norm(result)

        return result

"""
    ScaleConvLayer.
"""


import dgl.function as fn
import torch
import torch.nn as nn
from tools.utils_model import get_activation_func


class SocialScaleConv4x(nn.Module):

    def __init__(
        self,
        scales,
        size,
        activation,
        normalization,
        dropout=0.0
    ):

        super(SocialScaleConv4x, self).__init__()

        # Input parameters
        self._scales = scales
        self._size = size
        self._activation = activation
        self._normalization = normalization
        self._dropout = dropout

        # Callable Functions
        self._func_activation = get_activation_func(self._activation)

        # Layer parameters
        self._layer_linear = nn.Linear(
            self._size * 4, self._size * 4, bias=True)
        self._layer_dropout = nn.Dropout(self._dropout)

        if (self._normalization == 'batch'):
            self._layer_norm = nn.BatchNorm1d(self._size * 4)

        elif (self._normalization == 'layer'):
            self._layer_norm = nn.LayerNorm(
                self._size * 4, eps=1e-6, elementwise_affine=True)

        # Parameters initialization
        self.init_params(self._activation)

    @torch.no_grad()
    def init_params(self, activation):
        """Initialize layer parameters."""

        gain = nn.init.calculate_gain(activation)
        nn.init.xavier_uniform_(self._layer_linear.weight, gain=gain)
        nn.init.zeros_(self._layer_linear.bias)

    def message_func(self, nodes):
        """Custom message function."""

        return {'features': nodes.mailbox['message'].flatten(start_dim=1)}

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

        # Linear layer
        result = self._layer_linear(result)

        # Apply activation
        result = self._func_activation(result)

        # Apply normalization
        if self._normalization:
            result = self._layer_norm(result)

        return result

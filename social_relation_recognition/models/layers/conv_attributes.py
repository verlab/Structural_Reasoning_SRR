"""
    AttributesConvLayer.
"""


import dgl.nn as dglnn
import torch
import torch.nn as nn
from tools.utils_model import get_activation_func, get_aggregation_func


class AttributesConv(nn.Module):

    def __init__(
        self,
        scales,
        attributes,
        size,
        activation,
        aggregation,
        normalization,
        weights=True,
        bias=True,
        dropout=0.0
    ):

        super(AttributesConv, self).__init__()

        # Input parameters
        self._scales = scales
        self._attributes = attributes
        self._size = size
        self._activation = activation
        self._normalization = normalization
        self._weights = weights
        self._bias = bias
        self._dropout = dropout

        # Callable Functions
        self._func_activation = get_activation_func(self._activation)
        self._aggregation = aggregation if (aggregation != 'lse') else lse

        # Heteroconv Modules
        self._modules = {}

        for attribute in self._attributes:
            self._modules[attribute] = dglnn.GraphConv(
                self._size,
                self._size,
                norm='both',
                weight=self._weights,
                bias=self._bias,
                activation=self._func_activation,
                allow_zero_in_degree=False
            )

        # Layer parameters
        self.attribute_conv = dglnn.HeteroGraphConv(
            self._modules, aggregate=self._aggregation)
        self._layer_dropout = nn.Dropout(self._dropout)

        if (self._normalization == 'batch'):
            self._layer_norm = nn.BatchNorm1d(self._size)

        elif (self._normalization == 'layer'):
            self._layer_norm = nn.LayerNorm(
                self._size, eps=1e-6, elementwise_affine=True)

    def forward(self, graph, features):
        """Forward step."""

        # Check features
        for attribute in self._attributes:
            assert attribute in features, \
                '>> [ERROR] Inputs missing %s attribute features' % attribute

        # Apply dropout
        for attribute in self._attributes:
            features[attribute] = self._layer_dropout(features[attribute])

        # Get subgraph
        subgraph = graph.edge_type_subgraph(self._attributes)

        # Apply conv layers
        hidden_states = self.attribute_conv(subgraph, features)

        # Apply normalization
        if self._normalization:
            for state in hidden_states:
                hidden_states[state] = self._layer_norm(hidden_states[state])

        return hidden_states


def lse(tensors, dsttype):
    """Log-Sum-Exp function."""

    stacked = torch.stack(tensors, dim=0)

    return torch.logsumexp(stacked, dim=0)

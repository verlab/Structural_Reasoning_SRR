"""
    RelationConvLayer.
"""


import dgl.function as fn
import torch
import torch.nn as nn
from tools.utils_model import get_activation_func


class RelationConvFusion(nn.Module):

    def __init__(
        self,
        size,
        activation,
        normalization,
        residual=True,
        dropout=0.0
    ):

        super(RelationConvFusion, self).__init__()

        # Input parameters
        self._size = size
        self._activation = activation
        self._normalization = normalization
        self._residual = residual
        self._dropout = dropout

        # Callable Functions
        self._func_activation = get_activation_func(self._activation)

        # Layer parameters
        self._layer_W = nn.Linear(self._size, self._size, bias=True)
        self._layer_U = nn.Linear(self._size, self._size, bias=True)
        self._layer_V = nn.Linear(self._size, self._size, bias=True)
        self._attn_left = nn.Parameter(torch.FloatTensor(size=(1, self._size)))
        self._attn_right = nn.Parameter(
            torch.FloatTensor(size=(1, self._size)))
        self._layer_dropout = nn.Dropout(self._dropout)

        if (self._normalization == 'batch'):
            self._layer_norm = nn.BatchNorm1d(self._size)

        elif (self._normalization == 'layer'):
            self._layer_norm = nn.LayerNorm(
                self._size, eps=1e-6, elementwise_affine=True)

        # Parameters initialization
        self.init_params(self._activation)

    @torch.no_grad()
    def init_params(self, activation):
        """Initialize layer parameters."""

        gain = nn.init.calculate_gain(activation)
        nn.init.xavier_normal_(self._layer_W.weight, gain=gain)
        nn.init.zeros_(self._layer_W.bias)
        nn.init.xavier_normal_(self._layer_U.weight, gain=gain)
        nn.init.zeros_(self._layer_U.bias)
        nn.init.xavier_normal_(self._layer_V.weight, gain=gain)
        nn.init.zeros_(self._layer_V.bias)
        nn.init.xavier_normal_(self._attn_left, gain=gain)
        nn.init.xavier_normal_(self._attn_right, gain=gain)

    def forward(self, graph, features):
        """Forward step."""

        # Check the features
        assert 'relation' in features, \
            '>> [ERROR] Inputs missing relation features'

        # Get features
        result = features['relation']

        # If the graph has relation egdes
        if('relation' in graph.etypes):

            # Get subgraph and add features
            subgraph = graph['relation', 'relation', 'relation']

            # Apply feature dropout
            hidden_states = self._layer_dropout(features['relation'])

            # Message Passing
            Wh = self._layer_W(hidden_states)
            subgraph.ndata['Uh'] = self._layer_U(hidden_states)
            subgraph.ndata['Vh'] = self._layer_V(hidden_states)

            out_src = Wh * self._attn_left
            out_dst = Wh * self._attn_right

            subgraph.srcdata.update({'out_src': out_src})
            subgraph.dstdata.update({'out_dst': out_dst})

            subgraph.apply_edges(fn.u_add_v('out_src', 'out_dst', 'e'))
            subgraph.edata['sigma'] = torch.sigmoid(subgraph.edata['e'])
            subgraph.update_all(fn.u_mul_e('Vh', 'sigma', 'm'),
                                fn.sum('m', 'sum_sigma_h'))
            subgraph.update_all(fn.copy_e('sigma', 'm'),
                                fn.sum('m', 'sum_sigma'))
            subgraph.ndata['h'] = subgraph.ndata['Uh'] + \
                subgraph.ndata['sum_sigma_h'] / \
                (subgraph.ndata['sum_sigma'] + 1e-6)
            hidden_states = subgraph.ndata['h']

            # Apply norm
            if self._normalization:
                hidden_states = self._layer_norm(hidden_states)

            hidden_states = self._func_activation(hidden_states)

            # Apply residual
            if self._residual:
                result = result + hidden_states

            else:
                result = hidden_states

        return result

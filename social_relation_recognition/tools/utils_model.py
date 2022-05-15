"""
    Model related routines and configurations.
"""


import torch

"""Activation functions."""
functions_activation = [
    'elu',
    'leaky_relu',
    'relu',
    'selu',
    'sigmoid',
    'tanh'
]

"""Aggregation functions."""
functions_aggregation = [
    'lse',
    'max',
    'mean',
    'min',
    'sum'
]

"""Normalization functions."""
functions_normalization = [
    'batch',
    'layer'
]


def _lse_reduce_func(inputs, dim):
    """None."""

    return torch.logsumexp(inputs, dim=dim)


def _max_reduce_func(inputs, dim):
    """None."""

    return torch.max(inputs, dim=dim)[0]


def _mean_reduce_func(inputs, dim):
    """None."""

    return torch.mean(inputs, dim=dim)


def _min_reduce_func(inputs, dim):
    """None."""

    return torch.min(inputs, dim=dim)[0]


def _sum_reduce_func(inputs, dim):
    """None."""

    return torch.sum(inputs, dim=dim)


def get_aggregation_func(name):
    """None."""

    if name == 'lse':
        func = _lse_reduce_func

    elif name == 'max':
        func = _max_reduce_func

    elif name == 'mean':
        func = _mean_reduce_func

    elif name == 'min':
        func = _min_reduce_func

    elif name == 'sum':
        func = _sum_reduce_func

    else:
        raise KeyError("Unknown aggregation function: %s" % name)

    return func


def get_activation_func(name):
    """None."""

    if name == 'elu':
        func = torch.nn.functional.elu

    elif name == 'leaky_relu':
        func = torch.nn.functional.leaky_relu

    elif name == 'relu':
        func = torch.nn.functional.relu

    elif name == 'selu':
        func = torch.nn.functional.selu

    elif name == 'sigmoid':
        func = torch.sigmoid

    elif name == 'tanh':
        func = torch.tanh

    else:
        raise KeyError("Unknown activation function: %s" % name)

    return func


def _check_params_attributes(parameters, size):
    """
        'attributes': {
                        'conv': ...,
                        'size_input': ...,
                        'size_output': ...,
                        'activation': ...,
                        'aggregation': ...,
                        'dropout': ...
                    }
    """

    # Check conv method
    assert parameters['attributes']['conv'] in ['gcn'], \
        '>> [ERROR] Unknown attributes convolution method'

    # Check input size
    assert parameters['attributes']['size_input'] == size, \
        '>> [ERROR] Wrong attributes convolution output size'

    # Check output size
    assert parameters['attributes']['size_output'] == parameters['scales']['size_input'], \
        '>> [ERROR] Wrong attributes convolution output size'

    # Check activation function
    assert parameters['attributes']['activation'] in functions_activation, \
        '>> [ERROR] Unknown attributes convolution activation function'

    # Check aggregation function
    assert parameters['attributes']['aggregation'] in functions_aggregation, \
        '>> [ERROR] Unknown scales convolution aggregation function'


def _check_params_scales(parameters, size):
    """
        'scales': {
                        'conv': ...,
                        'size_input': ...,
                        'size_output': ...,
                        'activation': ...,
                        'aggregation': ...,
                        'dropout': ...
                    }
    """

    # Check conv method
    assert parameters['scales']['conv'] in ['3x', '4x'], \
        '>> [ERROR] Unknown scales convolution method'

    # Check input size
    assert parameters['scales']['size_input'] == size, \
        '>> [ERROR] Wrong scales convolution output size'

    # Check output size
    if (parameters['scales']['conv'] == '3x'):
        assert parameters['scales']['size_output'] == parameters['scales']['size_input'] * 3, \
            '>> [ERROR] Wrong scales convolution output size'

    else:
        assert parameters['scales']['size_output'] == parameters['scales']['size_input'] * 4, \
            '>> [ERROR] Wrong scales convolution output size'

    # Check activation function
    assert parameters['scales']['activation'] in functions_activation, \
        '>> [ERROR] Unknown scales convolution activation function'

    # Check aggregation function
    if (parameters['scales']['conv'] == '3x'):
        assert parameters['scales']['aggregation'] in functions_aggregation, \
            '>> [ERROR] Unknown scales convolution aggregation function'

    else:
        assert parameters['scales']['aggregation'] is None, \
            '>> [ERROR] Scales convolution aggregation function should be None'


def _check_params_relations(parameters):
    """
        'relations': {
                        'conv': ...,
                        'size_input': ...,
                        'size_output': ...,
                        'activation': ...,
                        'aggregation': ...,
                        'dropout': ...
                    }
    """

    # Check conv method
    assert parameters['relations']['conv'] in ['fusion', 'ggcn', 'gat'], \
        '>> [ERROR] Unknown relations convolution method'

    # Check input size
    assert parameters['relations']['size_input'] == parameters['scales']['size_output'], \
        '>> [ERROR] Wrong relations convolution input size'

    # Check output size
    assert parameters['relations']['size_output'] == parameters['relations']['size_input'], \
        '>> [ERROR] Wrong relations convolution output size'

    # Check activation function
    if (parameters['relations']['conv'] in ['fusion', 'ggnn']):
        assert parameters['relations']['activation'] in functions_activation, \
            '>> [ERROR] Unknown relations convolution activation function'

    else:
        assert parameters['relations']['activation'] is None, \
            '>> [ERROR] Relations convolution activation function should be None'

    # Check aggregation function
    if (parameters['relations']['conv'] in ['fusion', 'ggnn']):
        assert parameters['relations']['aggregation'] is None, \
            '>> [ERROR] Relations convolution aggregation function should be None'

    else:
        assert parameters['relations']['aggregation'] in functions_aggregation, \
            '>> [ERROR] Unknown relations convolution aggregation function'


def check_params(parameters, size, topology):
    """None."""

    # Check if attributes params were given
    if (topology == 'SKG'):
        assert parameters['attributes'] is None, \
            '>> [ERROR] Attribute parameters should be None'

    else:
        _check_params_attributes(parameters, size)

    _check_params_scales(parameters, size)
    _check_params_relations(parameters)

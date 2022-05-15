"""
    Social Graph Network.
"""


import torch
import torch.nn as nn
import torchvision.models as models
from tools.utils_dataset import feature2size

from models.layers.conv_attributes import AttributesConv
from models.layers.conv_relations_fusion import RelationConvFusion
from models.layers.conv_scales_3x import SocialScaleConv3x
from models.layers.conv_scales_4x import SocialScaleConv4x


class SocialRelationRecognition(nn.Module):

    def __init__(
        self,
        size,
        topology,
        scales,
        attributes,
        classes,
        conv_params,
        freeze_params=None
    ):

        super(SocialRelationRecognition, self).__init__()

        # Graph parameters
        self._size = size
        self._topology = topology
        self._scales = scales
        self._attributes = attributes
        self._classes = classes
        self._conv_params = conv_params

        # Layer parameters
        self._freeze_params = freeze_params

        # Social scale backbones
        if (self._topology != 'SKG-'):

            self._social_scale_backbones = nn.ModuleDict()

            for scale in self._scales:
                self._social_scale_backbones[scale] = models.resnet101(
                    pretrained=True)
                self._social_scale_backbones[scale].fc = nn.Identity()
                #self._social_scale_backbones[scale] = models.vgg19(pretrained=True)
                #self._social_scale_backbones[scale].classifier = self._social_scale_backbones[scale].classifier[:-1]

                # Freeze parameters
                if (self._freeze_params == 'all'):
                    for _, param in self._social_scale_backbones[scale].named_parameters():
                        param.requires_grad = False

                elif (self._freeze_params == 'conv'):
                    for name, param in self._social_scale_backbones[scale].named_parameters():
                        if("bn" not in name):
                            param.requires_grad = False

                elif (self._freeze_params == 'norm'):
                    for name, param in self._social_scale_backbones[scale].named_parameters():
                        if("bn" in name):
                            param.requires_grad = False

                elif (self._freeze_params == 'finetune'):
                    self._social_scale_backbones[scale].layer1.requires_grad_(
                        False)
                    self._social_scale_backbones[scale].layer2.requires_grad_(
                        False)

        # Input encoder layers
        self._encoder_layers = nn.ModuleDict()

        for scale in (self._scales + ['relation']):
            self._encoder_layers[scale] = nn.Sequential(
                nn.Linear(feature2size[scale], self._size),
                nn.ReLU(),
            )

        if (self._topology != 'SKG'):
            for attribute in self._attributes:
                self._encoder_layers[attribute] = nn.Sequential(
                    nn.Linear(feature2size[attribute], self._size),
                    nn.ReLU(),
                )

        # Attributes convolution layer
        if (self._topology != 'SKG'):
            if (self._conv_params['attributes']['conv'] == 'gcn'):
                self._conv_attributes = AttributesConv(
                    self._scales,
                    self._attributes,
                    self._conv_params['attributes']['size_input'],
                    activation=self._conv_params['attributes']['activation'],
                    aggregation=self._conv_params['attributes']['aggregation'],
                    normalization=self._conv_params['attributes']['normalization'],
                    dropout=self._conv_params['attributes']['dropout'],
                    weights=True, bias=True
                )

            else:
                raise KeyError("Unknown attributes convolution: %s" %
                               self._conv_params['attributes']['conv'])

        # Scales convolution layer
        if (self._conv_params['scales']['conv'] == '3x'):
            self._conv_scales = SocialScaleConv3x(
                self._scales,
                self._conv_params['scales']['size_input'],
                activation=self._conv_params['scales']['activation'],
                aggregation=self._conv_params['scales']['aggregation'],
                normalization=self._conv_params['scales']['normalization'],
                dropout=self._conv_params['scales']['dropout']
            )

        elif (self._conv_params['scales']['conv'] == '4x'):
            self._conv_scales = SocialScaleConv4x(
                self._scales,
                self._conv_params['scales']['size_input'],
                activation=self._conv_params['scales']['activation'],
                normalization=self._conv_params['scales']['normalization'],
                dropout=self._conv_params['scales']['dropout']
            )

        else:
            raise KeyError("Unknown scales convolution: %s" %
                           self._conv_params['scales']['conv'])

        # Relations convolution layer
        if (self._conv_params['relations']['conv'] == 'fusion'):
            self._conv_relations = RelationConvFusion(
                self._conv_params['relations']['size_input'],
                activation=self._conv_params['relations']['activation'],
                normalization=self._conv_params['relations']['normalization'],
                dropout=self._conv_params['relations']['dropout'],
                residual=True
            )

        else:
            raise KeyError("Unknown relations convolution: %s" %
                           self._conv_params['relations']['conv'])

        # Classifier layers
        self._classifier_layers = nn.ModuleDict()

        for classifier in ['scales', 'relations']:
            self._classifier_layers[classifier] = nn.Sequential(
                nn.Linear(self._conv_params['relations']['size_output'],
                          self._conv_params['relations']['size_output']),
                nn.ReLU(),
                nn.Linear(self._conv_params['relations']
                          ['size_output'], self._classes)
            )

        # Parameters initialization
        self._encoder_layers.apply(init_params)
        self._classifier_layers.apply(init_params)

    def forward(self, inputs, graph, features):
        """Forward step."""

        # Hidden features
        hidden_states = {}

        # Extract Social Scale features
        if (self._topology != 'SKG-'):
            for scale in self._scales:
                features[scale] = self._social_scale_backbones[scale](
                    inputs[scale]).view(-1, 2048)

        # Encode features
        for feature in features:
            hidden_states[feature] = self._encoder_layers[feature](
                features[feature])

        # Attributes conv
        if (self._topology != 'SKG'):
            outputs_attributes = self._conv_attributes(graph, hidden_states)

            for scale in outputs_attributes:
                hidden_states[scale] += outputs_attributes[scale]

        # Scales conv
        hidden_states['relation'] = self._conv_scales(graph, hidden_states)
        logits_scales = self._classifier_layers['scales'](
            hidden_states['relation'])

        # Relations conv
        hidden_states['relation'] = self._conv_relations(graph, hidden_states)
        logits_relations = self._classifier_layers['relations'](
            hidden_states['relation'])

        return logits_relations, logits_scales

    def get_params(self):
        """Return the parameters for each module."""

        params_social_scales = None

        if (self._topology != 'SKG-'):
            params_social_scales = [
                parameter for parameter in self._social_scale_backbones.parameters() if parameter.requires_grad]

        params_graph_network = (
            list(self._encoder_layers.parameters()) +
            list(self._conv_scales.parameters()) +
            list(self._conv_relations.parameters()) +
            list(self._classifier_layers.parameters())
        )

        if (self._topology != 'SKG'):
            params_graph_network += list(self._conv_attributes.parameters())

        return params_social_scales, params_graph_network


@torch.no_grad()
def init_params(module):
    """Initialize linear layer params."""

    gain = nn.init.calculate_gain('relu')

    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)

"""
    Inference for SSN and SGN models.
"""


import argparse
import os
import time

import cv2
import numpy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torchvision import transforms

from data.loader import get_test_loader
from models.sgn import SocialRelationRecognition
from tools.metrics import get_metrics
from tools.utils import (check_python_3, load_json_file, save_json_file,
                         set_environment_dgl_backend)
from tools.utils_dataset import (dataset_type2class, dataset_type2mean,
                                 dataset_type2std, dataset_type2weight,
                                 fix_bounding_box, get_bounding_boxes,
                                 get_context_bounding_box, splits)
from tools.utils_graph import build_graph, get_attributes, load_attributes
from tools.utils_pytorch import sanity_check

"""None."""
dataset2model = {
    'PIPA': '_best_a',
    'PISC': '_best_m'
}

"""None."""
type_inputs = [
    'personal_1',
    'personal_2',
    'local',
    'global'
]


class Tester(object):

    def __init__(self, args):

        # Metric type
        self.metric = args.metric

        # Paths
        self.path_configs = args.path_configs
        self.path_datasets = args.path_datasets
        self.path_models = args.path_models
        self.path_results = args.path_results

        # Modes
        self.mode_cuda = args.cuda
        self.mode_debug = args.debug
        self.mode_show = args.show
        self.mode_test = args.test

        # Information
        self.time_start = 0.
        self.time_total = 0.

    def load_configs(self):
        """Load the configurations from the YAML file."""

        with open(self.path_configs) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        self.configs = []

        # [RUN]

        self.name = data['Name']
        temp = 'Name: %s' % self.name
        self.configs.append(temp)

        self.dataset = data['Dataset']
        temp = 'Dataset: %s' % self.dataset
        self.configs.append(temp)

        self.relation = data['Relation Classes']
        temp = 'Relation Classes: %s' % self.relation
        self.configs.append(temp)

        self.loss = data['Final Loss']
        temp = 'Final Loss: %s' % self.loss
        self.configs.append(temp)

        self.statistics = data['Dataset Statistics']
        temp = 'Dataset Statistics: %s' % self.statistics
        self.configs.append(temp)

        self.freeze_params = data['Freeze Parameters']
        temp = 'Freeze Parameters: %s' % self.freeze_params
        self.configs.append(temp)

        self.use_weights = data['Class Weights']
        temp = 'Class Weights: %s' % self.use_weights
        self.configs.append(temp)

        self.use_amsgrad = data['Adam AMSGrad']
        temp = 'Adam AMSGrad: %s' % self.use_amsgrad
        self.configs.append(temp)

        self.seed = data['Random Seed']
        temp = 'Random Seed: %d' % self.seed
        self.configs.append(temp)

        # [GRAPH]

        self.topology = data['Graph Topology']
        temp = 'Graph Topology: %s' % self.topology
        self.configs.append(temp)

        self.scales = data['Social Scales']
        temp = 'Social Scales:'
        self.configs.append(temp)

        for scale in self.scales:
            temp = '  - %s' % scale
            self.configs.append(temp)

        self.edges = data['Relation Edges']
        temp = 'Relation Edges: %s' % self.edges
        self.configs.append(temp)

        # [MODEL]

        # Attributes
        temp = 'Attributes:'
        self.configs.append(temp)

        self.attributes_conv = data['Attributes'][0]['Convolution']
        temp = '  - Convolution: %s' % self.attributes_conv
        self.configs.append(temp)

        self.attributes_activation = data['Attributes'][1]['Activation']
        temp = '  - Activation: %s' % self.attributes_activation
        self.configs.append(temp)

        self.attributes_aggregation = data['Attributes'][2]['Aggregation']
        temp = '  - Aggregation: %s' % self.attributes_aggregation
        self.configs.append(temp)

        self.attributes_normalization = data['Attributes'][3]['Normalization']
        temp = '  - Normalization: %s' % self.attributes_normalization
        self.configs.append(temp)

        self.attributes_dropout = data['Attributes'][4]['Dropout']
        temp = '  - Dropout: %f' % self.attributes_dropout
        self.configs.append(temp)

        # Scales
        temp = 'Scales:'
        self.configs.append(temp)

        self.scales_conv = data['Scales'][0]['Convolution']
        temp = '  - Convolution: %s' % self.scales_conv
        self.configs.append(temp)

        self.scales_activation = data['Scales'][1]['Activation']
        temp = '  - Activation: %s' % self.scales_activation
        self.configs.append(temp)

        self.scales_aggregation = data['Scales'][2]['Aggregation']
        temp = '  - Aggregation: %s' % self.scales_aggregation
        self.configs.append(temp)

        self.scales_normalization = data['Scales'][3]['Normalization']
        temp = '  - Normalization: %s' % self.scales_normalization
        self.configs.append(temp)

        self.scales_dropout = data['Scales'][4]['Dropout']
        temp = '  - Dropout: %f' % self.scales_dropout
        self.configs.append(temp)

        # Relations
        temp = 'Relations:'
        self.configs.append(temp)

        self.relations_conv = data['Relations'][0]['Convolution']
        temp = '  - Convolution: %s' % self.relations_conv
        self.configs.append(temp)

        self.relations_activation = data['Relations'][1]['Activation']
        temp = '  - Activation: %s' % self.relations_activation
        self.configs.append(temp)

        self.relations_aggregation = data['Relations'][2]['Aggregation']
        temp = '  - Aggregation: %s' % self.relations_aggregation
        self.configs.append(temp)

        self.relations_normalization = data['Relations'][3]['Normalization']
        temp = '  - Normalization: %s' % self.relations_normalization
        self.configs.append(temp)

        self.relations_dropout = data['Relations'][4]['Dropout']
        temp = '  - Dropout: %f' % self.relations_dropout
        self.configs.append(temp)

        # [HYPERPARAMETERS]

        self.number_workers = data['Workers Number']
        temp = 'Workers Number: %d' % self.number_workers
        self.configs.append(temp)

        self.number_epochs = data['Epochs Number']
        temp = 'Epochs Number: %d' % self.number_epochs
        self.configs.append(temp)

        self.size_input = data['Input Size']
        temp = 'Input Size: %d' % self.size_input
        self.configs.append(temp)

        self.size_batch = data['Batch Size']
        temp = 'Batch Size: %d' % self.size_batch
        self.configs.append(temp)

        self.size_hidden = data['Hidden Size']
        temp = 'Hidden Size: %d' % self.size_hidden
        self.configs.append(temp)

        self.rate_learning = data['Learning Rate']
        temp = 'Learning Rate: %f' % self.rate_learning
        self.configs.append(temp)

        self.rate_regularization = data['Weight Regularization']
        temp = 'Weight Regularization: %f' % self.rate_regularization
        self.configs.append(temp)

        self.rate_momentum = data['Momentum']
        temp = 'Momentum: %f' % self.rate_momentum
        self.configs.append(temp)

        self.rate_decay = data['Decay Rate']
        temp = 'Decay Rate: %f' % self.rate_decay
        self.configs.append(temp)

        self.mode_eval = data['Eval Mode']
        temp = 'Eval Mode: %s' % self.mode_eval
        self.configs.append(temp)

        self.mode_save = data['Save Mode']
        temp = 'Save Mode: %s' % self.mode_save
        self.configs.append(temp)

    def set_conv_params(self):

        self.conv_params = {}
        size_output = self.size_hidden * \
            len(self.scales) if(self.scales_conv ==
                                '3x') else self.size_hidden * (len(self.scales) + 1)

        # Attribute conv parameters
        self.conv_params['attributes'] = {
            'conv': self.attributes_conv,
            'size_input': self.size_hidden,
            'size_output': self.size_hidden,
            'activation': self.attributes_activation,
            'aggregation': self.attributes_aggregation,
            'normalization': self.attributes_normalization,
            'dropout': self.attributes_dropout
        }

        # Scale conv parameters
        self.conv_params['scales'] = {
            'conv': self.scales_conv,
            'size_input': self.size_hidden,
            'size_output': size_output,
            'activation': self.scales_activation,
            'aggregation': self.scales_aggregation,
            'normalization': self.scales_normalization,
            'dropout': self.scales_dropout
        }

        # Relation conv parameters
        self.conv_params['relations'] = {
            'conv': self.relations_conv,
            'size_input': size_output,
            'size_output': size_output,
            'activation': self.relations_activation,
            'aggregation': self.relations_aggregation,
            'normalization': self.relations_normalization,
            'dropout': self.relations_dropout
        }

    def set_paths(self):
        """Set up the working paths for the model testing."""

        self.path_metadata = os.path.join(
            self.path_datasets, self.dataset, self.dataset + '.json')
        self.path_images = os.path.join(
            self.path_datasets, self.dataset, 'images', 'full')
        self.path_split = os.path.join(
            self.path_datasets, self.dataset, 'splits', self.relation, 'test.json')
        self.path_attributes = os.path.join(
            self.path_datasets, self.dataset, 'features', 'hdf5', self.relation, 'test')

        self.path_model = os.path.join(self.path_models, self.dataset, self.relation,
                                       self.topology, self.scales_conv, self.relations_conv, self.name)
        self.path_result = os.path.join(self.path_results, self.dataset, self.relation,
                                        self.topology, self.scales_conv, self.relations_conv, self.name)

        self.path_load = os.path.join(self.path_model, self.name)
        self.path_contextual = self.path_load + '_contextual_' + self.metric + '.pth'
        self.path_weights = self.path_load + '_weights_' + self.metric + '.pth'

    def set_info(self):
        """Set up classes and graph information."""

        # Set number of classes
        self.number_classes = dataset_type2class[self.dataset][self.relation]

        # Set node and edge types
        self.attributes = get_attributes(self.scales) if (
            self.topology != 'SKG') else None

    def set_data(self):
        """Set up data loaders and attribute features."""

        # Get metadata
        self.data_meta = load_json_file(self.path_metadata)

        print('>>   Setting dataloader...')

        self.data_split = load_json_file(self.path_split)
        self.data_loader = get_test_loader(self.dataset, self.path_images, self.data_split, self.data_meta,
                                           self.relation, self.size_input, self.size_batch, self.statistics, self.number_workers)

        print('>>   ...done!')

        if (self.topology != 'SKG'):

            print('>>   Loading attributes...')

            self.data_attributes = load_attributes(
                self.scales, self.path_attributes)

            print('>>   ...done!')

        else:
            self.data_attributes = None

    def set_device(self):
        """Set up the working device."""

        self.device = torch.device('cuda:0' if (
            self.mode_cuda and torch.cuda.is_available()) else 'cpu')

    def show_configs(self):
        """Print model configurations."""

        print('>> [CONFIGURATIONS]')

        for config in self.configs:
            print('>>   %s' % (config))

    def update_time(self):
        """Update total time meter."""

        self.time_total = time.time() - self.time_start

    def build_model(self):
        """Build the training model using the loaded configurations."""

        # Get class balance weights
        self.weights_class = torch.tensor(list(dataset_type2weight[self.dataset][self.relation].values(
        )), device=self.device) if self.use_weights else None

        # Get model and set to device
        self.model = SocialRelationRecognition(
            self.size_hidden,
            self.topology,
            self.scales,
            self.attributes,
            self.number_classes,
            self.conv_params,
            freeze_params=self.freeze_params
        )

        self.model.to(self.device)

    def load_checkpoint(self):
        """Load checkpoints."""

        print('>> [CHECKPOINT]')

        # Check paths
        assert os.path.isfile(self.path_contextual), \
            '>> [ERROR] No contextual checkpoint at: %s' % self.path_contextual

        assert os.path.isfile(self.path_contextual), \
            '>> [ERROR] No weights checkpoint at: %s' % self.path_weights

        # Load contextual
        print('>>   Loading contextual from: %s' % self.path_contextual)
        self.contextual = torch.load(self.path_contextual)
        print('>>   Current status:')
        print('>>     - Acc: %f' % self.contextual['acc'])
        print('>>     - mAP: %f' % self.contextual['map'])

        # Load weights
        print('>>   Loading weights from: %s' % self.path_weights)
        state_dict = torch.load(self.path_weights)
        self.model.load_state_dict(state_dict)

    def set_transformer(self):
        """Initialize transformer."""

        mean = [0.485, 0.456, 0.406] if (
            self.statistics == 'ImageNet') else dataset_type2mean[self.dataset][self.relation]
        std = [0.229, 0.224, 0.225] if (
            self.statistics == 'ImageNet') else dataset_type2std[self.dataset][self.relation]

        normalize = transforms.Normalize(mean=mean, std=std)

        self.transformer = transforms.Compose([
            transforms.Resize((self.size_input, self.size_input)),
            transforms.ToTensor(),
            normalize])

    def save_data(self, path, name, data, order=False):

        # Create folder
        if not os.path.isdir(path):
            os.makedirs(path)

        # Save json
        path_save_file = os.path.join(path, name + '.json')
        save_json_file(path_save_file, data, order)

    def test(self):
        """Test pipeline."""

        # Initialize results lists
        self.results_scales_predictions = {}
        self.results_relations_predictions = {}
        self.results_both_correct = {}
        self.results_scales_correct = {}
        self.results_relations_correct = {}

        # Initialize tester
        self.load_configs()
        self.set_paths()
        self.set_info()
        self.set_conv_params()
        self.set_data()
        self.set_device()
        self.set_transformer()

        # Show run information
        sanity_check()
        self.show_configs()

        # Build model and load weights
        self.build_model()
        self.load_checkpoint()

        # Path to results
        path_save = os.path.join(self.path_result, 'predictions')

        print('>> [TEST]')

        list_scores_scales = []
        list_losses_scales = []
        list_scores_relations = []
        list_losses_relations = []
        list_ids = []
        list_labels = []

        # Activate eval mode
        self.model.eval()

        # Turn of gradients
        with torch.no_grad():
            for (batch_ids, input_personal, input_local, input_global, batch_labels) in self.data_loader:

                # Get graph
                batch_graph, batch_features = build_graph(
                    self.scales,
                    self.topology,
                    self.edges,
                    self.data_meta,
                    self.relation,
                    batch_ids,
                    data_attributes=self.data_attributes
                )

                # Prepare inputs and send data to device
                batch_inputs = {}
                batch_graph = batch_graph.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if ('personal' in self.scales):
                    batch_inputs['personal'] = input_personal.to(self.device)

                if ('local' in self.scales):
                    batch_inputs['local'] = input_local.to(self.device)

                if ('global' in self.scales):
                    batch_inputs['global'] = input_global.to(self.device)

                for feature in batch_features:
                    batch_features[feature] = batch_features[feature].to(
                        self.device)

                # Forward step
                batch_logits_relations, batch_logits_scales = self.model(
                    batch_inputs, batch_graph, batch_features)

                # Get loss and class scores
                batch_losses_scales = F.cross_entropy(
                    batch_logits_scales, batch_labels, weight=self.weights_class)
                batch_scores_scales = F.softmax(batch_logits_scales, dim=1)

                batch_losses_relations = F.cross_entropy(
                    batch_logits_relations, batch_labels, weight=self.weights_class)
                batch_scores_relations = F.softmax(
                    batch_logits_relations, dim=1)

                list_losses_scales.append(batch_losses_scales.item())
                list_scores_scales.extend(batch_scores_scales.tolist())

                list_losses_relations.append(batch_losses_relations.item())
                list_scores_relations.extend(batch_scores_relations.tolist())

                list_ids.extend(batch_ids)
                list_labels.extend(batch_labels.tolist())

            full_losses_scales = numpy.asarray(list_losses_scales)
            full_scores_scales = numpy.asarray(list_scores_scales)
            full_losses_relations = numpy.asarray(list_losses_relations)
            full_scores_relations = numpy.asarray(list_scores_relations)
            full_labels = numpy.asarray(list_labels)

            # Get metrics and update meters
            metrics_scales = get_metrics(full_scores_scales, full_labels, [
                                         'accuracy', 'mean_average_precision', 'class_recall'])
            metrics_relations = get_metrics(full_scores_relations, full_labels, [
                                            'accuracy', 'mean_average_precision', 'class_recall'])

            # Get predictions
            predictions_scales = numpy.argmax(
                full_scores_scales, axis=1).tolist()
            predictions_relations = numpy.argmax(
                full_scores_relations, axis=1).tolist()

            # Show statistics
            print('>>   [Scales]\n>>     - Loss:{:.4f} - Acc:{:.4f} - mAP:{:.4f}'.
                  format(
                      numpy.mean(full_losses_scales),
                      metrics_scales['accuracy'],
                      metrics_scales['mean_average_precision']
                  ))
            print('>>     - Recall:', metrics_scales['class_recall'])

            print('>>   [Relations]\n>>     - Loss:{:.4f} - Acc:{:.4f} - mAP:{:.4f}'.
                  format(
                      numpy.mean(full_losses_relations),
                      metrics_relations['accuracy'],
                      metrics_relations['mean_average_precision']
                  ))
            print('>>     - Recall:', metrics_relations['class_recall'])

            # Counters
            counter_images = 0
            counter_relations = 0

            # Set results
            for id_image in self.data_split:

                # Test mode
                if self.mode_test:
                    assert list_ids[counter_images] == id_image, \
                        '>> [ERROR] ID missmatch for image %d' % id_image

                # Set new relations lists
                self.results_scales_predictions[id_image] = {}
                self.results_relations_predictions[id_image] = {}

                for id_relation in self.data_meta[id_image][self.relation]:

                    # Test mode
                    if self.mode_test:
                        assert list_labels[counter_relations] == self.data_meta[id_image][self.relation][id_relation], \
                            '>> [ERROR] Relationship label missmatch for image %d' % id_image

                    self.results_scales_predictions[id_image][id_relation] = predictions_scales[counter_relations]
                    self.results_relations_predictions[id_image][id_relation] = predictions_relations[counter_relations]

                    # Compare results
                    scales_correct = (
                        self.results_scales_predictions[id_image][id_relation] == self.data_meta[id_image][self.relation][id_relation])
                    relations_correct = (
                        self.results_relations_predictions[id_image][id_relation] == self.data_meta[id_image][self.relation][id_relation])
                    scales_diff_relations = (
                        self.results_scales_predictions[id_image][id_relation] != self.results_relations_predictions[id_image][id_relation])

                    if (scales_correct and relations_correct):
                        if not(id_image in self.results_both_correct):
                            self.results_both_correct[id_image] = []

                        self.results_both_correct[id_image].append(id_relation)

                    if (scales_correct and scales_diff_relations):
                        if not(id_image in self.results_scales_correct):
                            self.results_scales_correct[id_image] = []

                        self.results_scales_correct[id_image].append(
                            id_relation)

                    if (relations_correct and scales_diff_relations):
                        if not(id_image in self.results_relations_correct):
                            self.results_relations_correct[id_image] = []

                        self.results_relations_correct[id_image].append(
                            id_relation)

                    counter_relations += 1

                counter_images += 1

        # Save results
        self.save_data(path_save, 'predictions_scales',
                       self.results_scales_predictions)
        self.save_data(path_save, 'predictions_relations',
                       self.results_relations_predictions)
        self.save_data(path_save, 'corrects_both', self.results_both_correct)
        self.save_data(path_save, 'corrects_scales',
                       self.results_scales_correct)
        self.save_data(path_save, 'corrects_relations',
                       self.results_relations_correct)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SGN testing')
    parser.add_argument('metric', type=str, help='metric type to test', choices=[
        'acc',
        'map'
    ])
    parser.add_argument('path_configs', type=str,
                        help='path to configurations file')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('path_models', type=str, help='path to models')
    parser.add_argument('path_results', type=str, help='path to save results')
    parser.add_argument('--cuda', action='store_true',
                        help='activate cuda mode')
    parser.add_argument('--debug', action='store_true',
                        help='activate debug mode')
    parser.add_argument('--show', action='store_true',
                        help='activate show mode')
    parser.add_argument('--test', action='store_true',
                        help='activate test mode')

    args = parser.parse_args()

    # Check for python correct version and set DGL backend
    check_python_3()
    set_environment_dgl_backend('pytorch')

    tester = Tester(args)
    tester.test()

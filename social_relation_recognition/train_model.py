"""
    End-to-end training for the SSN and SGN models.
"""


import argparse
import math
import os
import random
import time

import dgl
import numpy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import tools.logger as logger
from data.checkpoint import Checkpoint
from data.loader import get_test_loader, get_train_loader
from models.sgn import SocialRelationRecognition
from models.weighted_loss import WeightedLoss
from tools.metrics import CumulativeMeter, get_metrics
from tools.utils import (check_python_3, get_time, load_json_file,
                         set_environment_dgl_backend)
from tools.utils_dataset import dataset_type2class, dataset_type2weight, splits
from tools.utils_graph import build_graph, get_attributes, load_attributes
from tools.utils_model import (functions_activation, functions_aggregation,
                               functions_normalization)
from tools.utils_pytorch import sanity_check


class Trainer(object):

    def __init__(self, args):

        # Run
        self.name = args.name
        self.dataset = args.dataset
        self.relation = args.relation
        self.loss = args.loss
        self.statistics = args.statistics
        self.freeze_params = None if (args.freeze == 'none') else args.freeze
        self.use_weights = True if (args.weights == 'true') else False
        self.use_amsgrad = True if (args.amsgrad == 'true') else False
        self.seed = args.seed if (args.seed >= 0) else random.randint(0, 999)

        # Graph
        self.topology = args.topology
        self.scales = args.scales
        self.edges = args.edges

        # Model
        # Attributes
        self.attributes_conv = None if (
            args.attributes_conv == 'none') else args.attributes_conv
        self.attributes_activation = None if (
            args.attributes_activation == 'none') else args.attributes_activation
        self.attributes_aggregation = None if (
            args.attributes_aggregation == 'none') else args.attributes_aggregation
        self.attributes_normalization = None if (
            args.attributes_normalization == 'none') else args.attributes_normalization
        self.attributes_dropout = args.attributes_dropout

        # Scales
        self.scales_conv = args.scales_conv
        self.scales_activation = None if (
            args.scales_activation == 'none') else args.scales_activation
        self.scales_aggregation = None if (
            args.scales_aggregation == 'none') else args.scales_aggregation
        self.scales_normalization = None if (
            args.scales_normalization == 'none') else args.scales_normalization
        self.scales_dropout = args.scales_dropout

        # Relations
        self.relations_conv = args.relations_conv
        self.relations_activation = None if (
            args.relations_activation == 'none') else args.relations_activation
        self.relations_aggregation = None if (
            args.relations_aggregation == 'none') else args.relations_aggregation
        self.relations_normalization = None if (
            args.relations_normalization == 'none') else args.relations_normalization
        self.relations_dropout = args.relations_dropout

        # Hyperparameters
        self.number_workers = args.workers
        self.number_epochs = args.epochs
        self.size_input = args.input
        self.size_batch = args.batch
        self.size_hidden = args.hidden
        self.rate_learning = args.learning
        self.rate_regularization = args.regularization
        self.rate_momentum = args.momentum
        self.rate_decay = args.decay

        # Paths
        self.path_datasets = args.path_datasets
        self.path_models = args.path_models
        self.path_results = args.path_results

        # Modes
        self.mode_eval = args.eval
        self.mode_save = args.save

        # Meters
        self.meter_epoch_time = CumulativeMeter()
        self.meter_epoch_loss = CumulativeMeter()
        self.meter_epoch_acc = CumulativeMeter()
        self.meter_train_loss = CumulativeMeter()
        self.meter_train_acc = CumulativeMeter()
        self.meter_test_acc_scales = CumulativeMeter()
        self.meter_test_loss_scales = CumulativeMeter()
        self.meter_test_map_scales = CumulativeMeter()
        self.meter_test_acc_relations = CumulativeMeter()
        self.meter_test_loss_relations = CumulativeMeter()
        self.meter_test_map_relations = CumulativeMeter()

        # Information
        self.time_start = 0.
        self.time_epoch = 0.
        self.time_total = 0.

    def set_configs(self):
        """Set up the YAML configuration file."""

        self.configs = []

        temp = '#[RUN]'
        self.configs.append(temp)

        temp = 'Name: %s' % self.name
        self.configs.append(temp)

        temp = 'Dataset: %s' % self.dataset
        self.configs.append(temp)

        temp = 'Relation Classes: %s' % self.relation
        self.configs.append(temp)

        temp = 'Final Loss: %s' % self.loss
        self.configs.append(temp)

        temp = 'Dataset Statistics: %s' % self.statistics
        self.configs.append(temp)

        temp = 'Freeze Parameters: %s' % ('null' if (
            self.freeze_params is None) else self.freeze_params)
        self.configs.append(temp)

        temp = 'Class Weights: %s' % str(self.use_weights).lower()
        self.configs.append(temp)

        temp = 'Adam AMSGrad: %s' % str(self.use_amsgrad).lower()
        self.configs.append(temp)

        temp = 'Random Seed: %d' % self.seed
        self.configs.append(temp)

        temp = '#[GRAPH]'
        self.configs.append(temp)

        temp = 'Graph Topology: %s' % self.topology
        self.configs.append(temp)

        temp = 'Social Scales:'
        self.configs.append(temp)

        for scale in self.scales:
            temp = '  - %s' % scale
            self.configs.append(temp)

        temp = 'Relation Edges: %s' % self.edges
        self.configs.append(temp)

        temp = '#[MODEL]'
        self.configs.append(temp)

        temp = 'Attributes:'
        self.configs.append(temp)

        temp = '  - Convolution: %s' % ('null' if (
            self.attributes_conv is None) else self.attributes_conv)
        self.configs.append(temp)

        temp = '  - Activation: %s' % ('null' if (
            self.attributes_activation is None) else self.attributes_activation)
        self.configs.append(temp)

        temp = '  - Aggregation: %s' % ('null' if (
            self.attributes_aggregation is None) else self.attributes_aggregation)
        self.configs.append(temp)

        temp = '  - Normalization: %s' % ('null' if (
            self.attributes_normalization is None) else self.attributes_normalization)
        self.configs.append(temp)

        temp = '  - Dropout: %f' % self.attributes_dropout
        self.configs.append(temp)

        temp = 'Scales:'
        self.configs.append(temp)

        temp = '  - Convolution: %s' % self.scales_conv
        self.configs.append(temp)

        temp = '  - Activation: %s' % ('null' if (
            self.scales_activation is None) else self.scales_activation)
        self.configs.append(temp)

        temp = '  - Aggregation: %s' % ('null' if (
            self.scales_aggregation is None) else self.scales_aggregation)
        self.configs.append(temp)

        temp = '  - Normalization: %s' % ('null' if (
            self.scales_normalization is None) else self.scales_normalization)
        self.configs.append(temp)

        temp = '  - Dropout: %f' % self.scales_dropout
        self.configs.append(temp)

        temp = 'Relations:'
        self.configs.append(temp)

        temp = '  - Convolution: %s' % self.relations_conv
        self.configs.append(temp)

        temp = '  - Activation: %s' % ('null' if (
            self.relations_activation is None) else self.relations_activation)
        self.configs.append(temp)

        temp = '  - Aggregation: %s' % ('null' if (
            self.relations_aggregation is None) else self.relations_aggregation)
        self.configs.append(temp)

        temp = '  - Normalization: %s' % ('null' if (
            self.relations_normalization is None) else self.relations_normalization)
        self.configs.append(temp)

        temp = '  - Dropout: %f' % self.relations_dropout
        self.configs.append(temp)

        temp = '#[HYPERPARAMETERS]'
        self.configs.append(temp)

        temp = 'Workers Number: %d' % self.number_workers
        self.configs.append(temp)

        temp = 'Epochs Number: %d' % self.number_epochs
        self.configs.append(temp)

        temp = 'Input Size: %d' % self.size_input
        self.configs.append(temp)

        temp = 'Batch Size: %d' % self.size_batch
        self.configs.append(temp)

        temp = 'Hidden Size: %d' % self.size_hidden
        self.configs.append(temp)

        temp = 'Learning Rate: %f' % self.rate_learning
        self.configs.append(temp)

        temp = 'Weight Regularization: %f' % self.rate_regularization
        self.configs.append(temp)

        temp = 'Momentum: %f' % self.rate_momentum
        self.configs.append(temp)

        temp = 'Decay Rate: %f' % self.rate_decay
        self.configs.append(temp)

        temp = 'Eval Mode: %s' % str(self.mode_eval).lower()
        self.configs.append(temp)

        temp = 'Save Mode: %s' % str(self.mode_save).lower()
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
        """Set up the working paths for the model training."""

        self.path_metadata = os.path.join(
            self.path_datasets, self.dataset, self.dataset + '.json')
        self.path_images = os.path.join(
            self.path_datasets, self.dataset, 'images', 'full')
        self.path_splits = {}
        self.path_attributes = {}

        for split in splits:
            self.path_splits[split] = os.path.join(
                self.path_datasets, self.dataset, 'splits', self.relation, split + '.json')
            self.path_attributes[split] = os.path.join(
                self.path_datasets, self.dataset, 'features', 'hdf5', self.relation, split)

        self.path_model = os.path.join(self.path_models, self.dataset, self.relation,
                                       self.topology, self.scales_conv, self.relations_conv, self.name)
        self.path_result = os.path.join(self.path_results, self.dataset, self.relation,
                                        self.topology, self.scales_conv, self.relations_conv, self.name)

        if not os.path.isdir(self.path_model):
            os.makedirs(self.path_model)

        if not os.path.isdir(self.path_result):
            os.makedirs(self.path_result)

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
        self.data_splits = {}
        self.data_loaders = {}
        self.data_attributes = {}

        print('>>   Setting dataloaders...')

        for split in splits:

            if (split == 'validation'):
                continue

            self.data_splits[split] = load_json_file(self.path_splits[split])
            self.data_attributes[split] = None

        self.data_loaders['train'] = get_train_loader(self.dataset, self.path_images, self.data_splits['train'],
                                                      self.data_meta, self.relation, self.size_input, self.size_batch, self.statistics, self.number_workers)
        self.data_loaders['test'] = get_test_loader(self.dataset, self.path_images, self.data_splits['test'],
                                                    self.data_meta, self.relation, self.size_input, self.size_batch, self.statistics, self.number_workers)

        print('>>   ...done!')

        if (self.topology != 'SKG'):

            print('>>   Loading attributes...')

            for split in splits:

                if (split == 'validation'):
                    continue

                self.data_attributes[split] = load_attributes(
                    self.scales, self.path_attributes[split])

            print('>>   ...done!')

    def set_recorder(self):
        """Set up the model checkpoint recorder."""

        self.recorder = Checkpoint(self.path_model, self.name)

    def set_logger(self):
        """Set up the configuration and score logger."""

        path_logger = os.path.join(self.path_result, 'logs')

        if not os.path.isdir(path_logger):
            os.makedirs(path_logger)

        self.logger = logger.Logger(self.name, path_logger, 'train')

    def set_writer(self):
        """Set up the Tensorboard writer."""

        path_writer = os.path.join(self.path_result, 'summaries')

        if not os.path.isdir(path_writer):
            os.makedirs(path_writer)

        self.writer = SummaryWriter(path_writer)

    def set_device(self):
        """Set up the working device."""

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_seed(self):
        """Set up the random seeds for reproducibility."""

        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        dgl.seed(self.seed)
        dgl.random.seed(self.seed)

        # torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def show_configs(self):
        """Print model configurations."""

        print('>> [CONFIGURATIONS]')

        for config in self.configs:
            print('>>   %s' % (config))

    def reset_meters(self):
        """Reset epoch meters."""

        self.meter_epoch_loss.reset()
        self.meter_epoch_acc.reset()

        self.time_epoch = time.time()

    def update_time(self):
        """Update total time meter."""

        self.time_total = time.time() - self.time_start

    def update_meters(self, epoch, batch, step, loss, acc):
        """Update epoch and total meters."""

        self.meter_epoch_loss.update(epoch, batch, step, loss)
        self.meter_epoch_acc.update(epoch, batch, step, acc)

        self.meter_train_loss.update(epoch, batch, step, loss)
        self.meter_train_acc.update(epoch, batch, step, acc)

    def update_summary(self, global_step):
        """Update Tensorboard summary writers."""

        self.writer.add_scalar(
            'Test/Scales/Loss', self.meter_test_loss_scales.value, global_step=global_step)
        self.writer.add_scalar(
            'Test/Scales/Accuracy', self.meter_test_acc_scales.value, global_step=global_step)
        self.writer.add_scalar(
            'Test/Scales/mAP', self.meter_test_map_scales.value, global_step=global_step)

        self.writer.add_scalar(
            'Test/Relations/Loss', self.meter_test_loss_relations.value, global_step=global_step)
        self.writer.add_scalar(
            'Test/Relations/Accuracy', self.meter_test_acc_relations.value, global_step=global_step)
        self.writer.add_scalar(
            'Test/Relations/mAP', self.meter_test_map_relations.value, global_step=global_step)

        self.writer.flush()

    def log_and_save(self, epoch):
        """Log and save time and best scores statistics."""

        message_loss_scales = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_loss_scales.minimum['epoch'],
            self.meter_test_loss_scales.minimum['batch'],
            self.meter_test_loss_scales.minimum['step'],
            self.meter_test_loss_scales.minimum['value']
        )

        message_acc_scales = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_acc_scales.maximum['epoch'],
            self.meter_test_acc_scales.maximum['batch'],
            self.meter_test_acc_scales.maximum['step'],
            self.meter_test_acc_scales.maximum['value']
        )

        message_map_scales = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_map_scales.maximum['epoch'],
            self.meter_test_map_scales.maximum['batch'],
            self.meter_test_map_scales.maximum['step'],
            self.meter_test_map_scales.maximum['value']
        )

        message_loss_relations = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_loss_relations.minimum['epoch'],
            self.meter_test_loss_relations.minimum['batch'],
            self.meter_test_loss_relations.minimum['step'],
            self.meter_test_loss_relations.minimum['value']
        )

        message_acc_relations = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_acc_relations.maximum['epoch'],
            self.meter_test_acc_relations.maximum['batch'],
            self.meter_test_acc_relations.maximum['step'],
            self.meter_test_acc_relations.maximum['value']
        )

        message_map_relations = '    - Epoch: %d\n    - Batch: %d\n    - Step: %d\n    - Value: %f' % (
            self.meter_test_map_relations.maximum['epoch'],
            self.meter_test_map_relations.maximum['batch'],
            self.meter_test_map_relations.maximum['step'],
            self.meter_test_map_relations.maximum['value']
        )

        self.update_time()
        str_time = get_time(self.time_total)
        self.logger.log_metric('scales', 'loss', message_loss_scales)
        self.logger.log_metric('scales', 'acc', message_acc_scales)
        self.logger.log_metric('scales', 'map', message_map_scales)
        self.logger.log_metric('relations', 'loss', message_loss_relations)
        self.logger.log_metric('relations', 'acc', message_acc_relations)
        self.logger.log_metric('relations', 'map', message_map_relations)
        self.logger.log_epoch(str(epoch))
        self.logger.log_time(str_time)
        self.logger.save()

        if self.mode_save:
            self.recorder.record_contextual({
                'acc': self.meter_test_acc_relations.value,
                'map': self.meter_test_map_relations.value
            })

            self.recorder.save_checkpoint(self.model)

    def build_model(self):
        """Build the training model using the input configurations."""

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

        # Get learnable parameter lists
        params_social_scales, params_graph_network = self.model.get_params()

        # Get class balance weights
        self.weights_class = torch.tensor(list(dataset_type2weight[self.dataset][self.relation].values(
        )), device=self.device) if self.use_weights else None

        # Get loss
        self.criterion = WeightedLoss(self.weights_class, self.loss)
        self.criterion.to(self.device)
        params_graph_network += list(self.criterion.parameters())

        # Set optimizers and schedulers
        self.optimizers_schedulers = []

        if (self.topology != 'SKG-'):
            optimizer_ssn = torch.optim.SGD(
                params_social_scales,
                lr=self.rate_learning/2,
                weight_decay=self.rate_regularization*5,
                momentum=self.rate_momentum
            )

            scheduler_ssn = StepLR(
                optimizer_ssn, step_size=1, gamma=self.rate_decay)
            self.optimizers_schedulers.append(
                ('SSN', optimizer_ssn, scheduler_ssn))

        optimizer_sgn = torch.optim.Adam(
            params_graph_network,
            lr=self.rate_learning,
            weight_decay=self.rate_regularization,
            amsgrad=self.use_amsgrad
        )

        scheduler_sgn = StepLR(
            optimizer_sgn, step_size=1, gamma=self.rate_decay)
        self.optimizers_schedulers.append(
            ('SGN', optimizer_sgn, scheduler_sgn))

    def load_checkpoint(self):
        """Load best checkpoint."""

        model_type = 'map' if (self.dataset == 'PISC') else 'acc'

        self.recorder.load_checkpoint(self.model, model_type)

    def test(self, epoch, batch, step):
        """Test pipeline."""

        print('>> [TEST]')

        list_scores_scales = []
        list_losses_scales = []
        list_scores_relations = []
        list_losses_relations = []
        list_labels = []

        # Activate eval mode
        self.model.eval()

        # Turn of gradients
        with torch.no_grad():
            for (batch_ids, input_personal, input_local, input_global, batch_labels) in self.data_loaders['test']:

                # Get graph
                batch_graph, batch_features = build_graph(
                    self.scales,
                    self.topology,
                    self.edges,
                    self.data_meta,
                    self.relation,
                    batch_ids,
                    data_attributes=self.data_attributes['test']
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

            self.meter_test_loss_scales.update(epoch, batch, step, numpy.mean(
                full_losses_scales), len(self.data_loaders['test'].dataset))
            self.meter_test_acc_scales.update(
                epoch, batch, step, metrics_scales['accuracy'])
            self.meter_test_map_scales.update(
                epoch, batch, step, metrics_scales['mean_average_precision'])
            self.meter_test_loss_relations.update(epoch, batch, step, numpy.mean(
                full_losses_relations), len(self.data_loaders['test'].dataset))
            self.meter_test_acc_relations.update(
                epoch, batch, step, metrics_relations['accuracy'])
            self.meter_test_map_relations.update(
                epoch, batch, step, metrics_relations['mean_average_precision'])

            # Show statistics
            print('>>   [Epoch:{:05d}|Batch:{:05d}|Step:{:05d}]'.format(
                epoch, batch, step))

            if (self.loss == 'weighted'):
                print('>>     [Weighted Loss]: %f' %
                      self.criterion.weights_loss.item())

            print('>>     [Scales]\n>>       - Loss:{:.4f} ({:.4f}) - Acc:{:.4f} ({:.4f}) - mAP:{:.4f} ({:.4f})'.
                  format(
                      self.meter_test_loss_scales.value, self.meter_test_loss_scales.minimum[
                          'value'],
                      self.meter_test_acc_scales.value, self.meter_test_acc_scales.maximum[
                          'value'],
                      self.meter_test_map_scales.value, self.meter_test_map_scales.maximum['value']
                  ))
            print('>>       - Recall:', metrics_scales['class_recall'])

            print('>>     [Relations]\n>>       - Loss:{:.4f} ({:.4f}) - Acc:{:.4f} ({:.4f}) - mAP:{:.4f} ({:.4f})'.
                  format(
                      self.meter_test_loss_relations.value, self.meter_test_loss_relations.minimum[
                          'value'],
                      self.meter_test_acc_relations.value, self.meter_test_acc_relations.maximum[
                          'value'],
                      self.meter_test_map_relations.value, self.meter_test_map_relations.maximum[
                          'value']
                  ))
            print('>>       - Recall:', metrics_relations['class_recall'])

        # Go back to train mode
        self.model.train()

        return

    def train(self):
        """Train pipeline."""

        # Initialize trainer
        self.set_seed()
        self.set_configs()
        self.set_paths()
        self.set_info()
        self.set_conv_params()
        self.set_data()
        self.set_recorder()
        self.set_logger()
        self.set_writer()
        self.set_device()

        # Show run information
        sanity_check()
        self.show_configs()
        self.logger.log_configs(self.configs)

        # Build model and load checkpoints
        self.build_model()
        self.load_checkpoint()

        # Use ImageNet running estimates
        if (self.mode_eval) and (self.topology != 'SKG-'):
            self.model._social_scale_backbones.eval()

        # Get sizes
        train_size = len(self.data_loaders['train'].dataset)
        batch_size = self.size_batch if ((self.size_batch > 0) and (
            self.size_batch <= train_size)) else train_size

        # Print and test frequency
        fraction_1_percent = train_size * 0.01
        fraction_10_percent = train_size * 0.1

        # Training loop start timer
        self.time_start = time.time()

        counter_global = 0

        # Epoch loop
        for counter_epoch in range(self.number_epochs + 1):

            print('>> [TRAIN]')

            # Reset epoch meters and counters
            self.reset_meters()

            counter_print = fraction_1_percent
            counter_test = fraction_10_percent
            counter_step = 0

            # Batch loop
            for counter_batch, (batch_ids, input_personal, input_local, input_global, batch_labels) in enumerate(self.data_loaders['train']):

                # Update step counter
                counter_step = (counter_batch + 1) * batch_size if (
                    ((counter_batch + 1) * batch_size) < train_size) else train_size

                # Get graph
                batch_graph, batch_features = build_graph(
                    self.scales,
                    self.topology,
                    self.edges,
                    self.data_meta,
                    self.relation,
                    batch_ids,
                    data_attributes=self.data_attributes['train']
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

                # Compute loss
                batch_losses_final = self.criterion(
                    batch_logits_relations, batch_logits_scales, batch_labels)

                # Backward step
                for triple in self.optimizers_schedulers:
                    _, optimizer, _ = triple
                    optimizer.zero_grad()

                batch_losses_final.backward()

                for triple in self.optimizers_schedulers:
                    _, optimizer, _ = triple
                    optimizer.step()

                # Show every 1%
                if (counter_print <= counter_step):

                    # Update print counter
                    counter_print = (int(counter_step/counter_print)
                                     * counter_print) + fraction_1_percent

                    # Get metrics
                    metrics_relations = get_metrics(
                        batch_logits_relations.tolist(), batch_labels.tolist(), ['accuracy'])

                    # Update meters
                    self.update_meters(counter_epoch, (counter_batch + 1), counter_step,
                                       batch_losses_final.item(), metrics_relations['accuracy'])

                    # Print statistics
                    print('>>   [Epoch:{:05d}|Batch:{:05d}|Step:{:05d}]'.format(
                        counter_epoch, (counter_batch + 1), counter_step), end=" ")

                    for triple in self.optimizers_schedulers:
                        name, _, scheduler = triple
                        print('LR({}):{:.5f}'.format(
                            name, scheduler.get_last_lr()[0]), end=" - ")

                    print('Loss:{:.4f} ({:.4f}) - Acc:{:.4f} ({:.4f})'.format(
                        self.meter_epoch_loss.average, self.meter_train_loss.average,
                        self.meter_epoch_acc.average, self.meter_train_acc.average,
                    ))

                # Test every 10%
                if (counter_test <= counter_step):

                    # Update print counter
                    counter_test = (int(counter_step/counter_test)
                                    * counter_test) + fraction_10_percent

                    # Run test
                    self.test(counter_epoch, counter_batch + 1, counter_step)

                    # Use ImageNet running estimates
                    if (self.mode_eval) and (self.topology != 'SKG-'):
                        self.model._social_scale_backbones.eval()

                    # Update meters and save
                    counter_global = (
                        counter_epoch * train_size) + counter_step
                    self.log_and_save(counter_epoch)
                    self.update_summary(counter_global)

                    if (counter_step < train_size):
                        print('>> [TRAIN]')

            # Update timers
            self.update_time()
            self.meter_epoch_time.update(counter_epoch, math.ceil(
                train_size/batch_size), counter_step, time.time() - self.time_epoch)

            # Schedulers step
            for triple in self.optimizers_schedulers:
                _, _, scheduler = triple
                scheduler.step()

            # Show every epoch
            print('>> [TIME]')
            print('>>   - Total: {}'.format(get_time(self.time_total)))
            print('>>   - Epoch: {} ({})'.format(get_time(self.meter_epoch_time.value),
                  get_time(self.meter_epoch_time.average)))

        self.writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SGN training')
    ###########################
    ##          RUN          ##
    ###########################
    parser.add_argument('name', type=str, help='run name')
    parser.add_argument('dataset', type=str, help='dataset to train on', choices=[
        'PIPA',
        'PISC'
    ])
    parser.add_argument('relation', type=str, help='type of relation to classify', choices=[
        'domain',
        'relationship'
    ])
    parser.add_argument('loss', type=str, help='final loss', choices=[
        'sum',
        'weighted'
    ])
    parser.add_argument('statistics', type=str, help='dataset statistics', choices=[
        'ImageNet',
        'Self'
    ])
    parser.add_argument('freeze', type=str, help='backbone parameters to freeze', choices=[
        'all',
        'conv',
        'finetune',
        'norm',
        'none'
    ])
    parser.add_argument('weights', type=str, help='use class balance weights', choices=[
        'false',
        'true'
    ])
    parser.add_argument('amsgrad', type=str, help='use amsgrad on AdamW', choices=[
        'false',
        'true'
    ])
    parser.add_argument(
        'seed', type=int, help='seed for reproducibility [-1: random]')
    ###########################
    ##         GRAPH         ##
    ###########################
    parser.add_argument('topology', type=str, help='graph topology version', choices=[
        'SKG-',
        'SKG',
        'SKG+'
    ])
    parser.add_argument('scales', type=str, nargs='+', help='social scales to extract information', choices=[
        'personal',
        'local',
        'global'
    ])
    parser.add_argument('edges', type=str, help='type of relation connection edge', choices=[
        'neighbors',
        'full',
        'none'
    ])
    ###########################
    ##         MODEL         ##
    ###########################
    # Attributes
    parser.add_argument('attributes_conv', type=str, help='attributes convolution type', choices=[
        'gcn',
        'none'
    ])
    parser.add_argument('attributes_activation', type=str,
                        help='attributes convolution activation', choices=functions_activation+['none'])
    parser.add_argument('attributes_aggregation', type=str,
                        help='attributes convolution aggregation', choices=functions_aggregation+['none'])
    parser.add_argument('attributes_normalization', type=str,
                        help='attributes convolution dropout rate', choices=functions_normalization+['none'])
    parser.add_argument('attributes_dropout', type=float,
                        help='attributes convolution dropout rate')
    # Scales
    parser.add_argument('scales_conv', type=str, help='scales convolution type', choices=[
        '3x',
        '4x'
    ])
    parser.add_argument('scales_activation', type=str,
                        help='scales convolution activation', choices=functions_activation+['none'])
    parser.add_argument('scales_aggregation', type=str,
                        help='scales convolution aggregation', choices=functions_aggregation+['none'])
    parser.add_argument('scales_normalization', type=str,
                        help='scales convolution dropout rate', choices=functions_normalization+['none'])
    parser.add_argument('scales_dropout', type=float,
                        help='scales convolution dropout rate')
    # Relations
    parser.add_argument('relations_conv', type=str, help='relations convolution type', choices=[
        'fusion',
        'gat',
        'ggcn'
    ])
    parser.add_argument('relations_activation', type=str,
                        help='relations convolution activation', choices=functions_activation+['none'])
    parser.add_argument('relations_aggregation', type=str,
                        help='relations convolution aggregation', choices=functions_aggregation+['none'])
    parser.add_argument('relations_normalization', type=str,
                        help='relations convolution dropout rate', choices=functions_normalization+['none'])
    parser.add_argument('relations_dropout', type=float,
                        help='relations convolution dropout rate')
    ###########################
    ##    HYPERPARAMETERS    ##
    ###########################
    parser.add_argument('workers', type=int, help='number of loder workers')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('input', type=int, help='input size')
    parser.add_argument('batch', type=int,
                        help='size of the batch [-1: full batch]')
    parser.add_argument('hidden', type=int, help='hidden state dimension')
    parser.add_argument('learning', type=float, help='initial learning rate')
    parser.add_argument('regularization', type=float,
                        help='weight regularization for the optimizers')
    parser.add_argument('momentum', type=float,
                        help='momentum for the SGD optimizer')
    parser.add_argument(
        'decay', type=float, help='decay rate for learning rate and weight regularization')
    ###########################
    ##         PATHS         ##
    ###########################
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('path_models', type=str,
                        help='path to save or load models')
    parser.add_argument('path_results', type=str,
                        help='path to save results data')
    ###########################
    ##         MODES         ##
    ###########################
    parser.add_argument('--debug', default=False,
                        action='store_true', help='activate debug mode')
    parser.add_argument('--eval', default=False,
                        action='store_true', help='activate eval mode')
    parser.add_argument('--save', default=False,
                        action='store_true', help='activate save mode')

    args = parser.parse_args()

    # Check for python correct version and set DGL backend
    check_python_3()
    set_environment_dgl_backend('pytorch')

    trainer = Trainer(args)
    trainer.train()

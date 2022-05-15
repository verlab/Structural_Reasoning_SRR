"""
	Class to save and load models.
"""


import os

import torch


class Checkpoint:

    def __init__(self, path, name):

        self._path = path
        self._name = name

        self.contextual = {
            'acc': 0.,
            'map': 0.
        }

        self._best_acc = 0
        self._best_map = 0

        self._save_acc = False
        self._save_map = False

    def record_contextual(self, contextual):
        """Record current and best metrics."""

        self.contextual = contextual

        if self.contextual['acc'] > self._best_acc:

            self._save_acc = True
            self._best_acc = self.contextual['acc']

        else:
            self._save_acc = False

        if self.contextual['map'] > self._best_map:

            self._save_map = True
            self._best_map = self.contextual['map']

        else:
            self._save_map = False

    def save_checkpoint(self, model):
        """Save model and contextual."""

        if (self._save_acc or self._save_map):
            print('>> [CHECKPOINT]')

            # Get paths
            path_save = os.path.join(self._path, self._name)

            if self._save_acc:

                torch.save(model.state_dict(), path_save + '_weights_acc.pth')
                torch.save(self.contextual, path_save + '_contextual_acc.pth')
                print('>>   Best accuracy weights and contextual saved')

            if self._save_map:

                torch.save(model.state_dict(), path_save + '_weights_map.pth')
                torch.save(self.contextual, path_save + '_contextual_map.pth')
                print('>>   Best mAP weights and contextual saved')

    def load_checkpoint(self, model, type):
        """Load model and contextual."""

        print('>> [CHECKPOINT]')

        # Check model type
        assert type in ['acc', 'map'], \
            '>> [ERROR] Unknown model type: %s' % type

        # Get paths
        path_load = os.path.join(self._path, self._name)
        path_contextual = path_load + '_contextual_' + type + '.pth'
        path_weights = path_load + '_weights_' + type + '.pth'

        # Load contextual
        if (path_load and os.path.isfile(path_contextual)):

            print('>>   Loading contextual from: %s' % path_contextual)
            self.contextual = torch.load(path_contextual)
            print('>>   Current status:')
            print('>>     - Acc: %f' % self.contextual['acc'])
            print('>>     - mAP: %f' % self.contextual['map'])

        else:
            print('>>   No contextual checkpoint at: %s' % path_contextual)

        # Load weights
        if path_load and os.path.isfile(path_weights):

            print('>>   Loading weights from: %s' % path_weights)
            state_dict = torch.load(path_weights)
            model.load_state_dict(state_dict)

        else:
            print('>>   No weights checkpoint at: %s' % path_weights)

"""
    Custom logger to save initial and final configurations for dataset processing, feature extraction and network training.
"""


import os


class Logger(object):

    def __init__(self, name, save_path, type):

        self.type_list = ['train']
        self.modules = ['scales', 'relations']
        self.metrics = ['loss', 'acc', 'map']

        assert isinstance(name, str), \
            '>> [LOGGER] Name is not a string'

        assert os.path.isdir(save_path), \
            '>> [LOGGER] Save path is not a directory'

        assert type in self.type_list, \
            '>> [LOGGER] Wrong logger type'

        # Logger Configs
        self.name = name
        self.save_path = save_path
        self.type = type

        # Logger Data
        self.configs = []

        self.total_epochs = 'Not logged yet'
        self.total_time = 'Not logged yet'

        self.best_scores = {
            'scales': {
                'loss': '    - Not logged yet',
                'acc': '    - Not logged yet',
                'map': '    - Not logged yet'
            },
            'relations': {
                'loss': '    - Not logged yet',
                'acc': '    - Not logged yet',
                'map': '    - Not logged yet'
            }
        }

        print('>> [LOGGER] Creating log on: ' + self.save_path)

    def log_configs(self, configurations):

        assert self.type == 'train', \
            '>> [LOGGER] Wrong logger type'

        self.configs = configurations

    def log_metric(self, module, metric, value):

        assert self.type == 'train', \
            '>> [LOGGER] Wrong logger type'

        assert module in self.modules, \
            '>> [LOGGER] Invalid module'

        assert metric in self.metrics, \
            '>> [LOGGER] Invalid metric'

        self.best_scores[module][metric] = value

    def log_epoch(self, epochs):

        assert self.type == 'train', \
            '>> [LOGGER] Wrong logger type'

        self.total_epochs = epochs

    def log_time(self, time):

        assert self.type == 'train', \
            '>> [LOGGER] Wrong logger type'

        self.total_time = time

    def save(self):

        if (self.type == 'train'):

            path_configs = os.path.join(
                self.save_path, 'configs_' + self.name + '.yaml')

            with open(path_configs, 'w') as file:
                for conf in self.configs:
                    file.write(conf + '\n')

            path_scores = os.path.join(
                self.save_path, 'scores_' + self.name + '.yaml')

            with open(path_scores, 'w') as file:
                file.write('#[TRAINING]\n')
                file.write('Epochs: ' + self.total_epochs + '\n')
                file.write('Time: ' + self.total_time + '\n')
                file.write('#[BEST SCORES]\n')
                file.write('Scales:\n')
                file.write('  - Loss:\n')
                file.write(self.best_scores['scales']['loss'] + '\n')
                file.write('  - Accuracy:\n')
                file.write(self.best_scores['scales']['acc'] + '\n')
                file.write('  - mAP:\n')
                file.write(self.best_scores['scales']['map'] + '\n')
                file.write('Relations:\n')
                file.write('  - Loss:\n')
                file.write(self.best_scores['relations']['loss'] + '\n')
                file.write('  - Accuracy:\n')
                file.write(self.best_scores['relations']['acc'] + '\n')
                file.write('  - mAP:\n')
                file.write(self.best_scores['relations']['map'] + '\n')

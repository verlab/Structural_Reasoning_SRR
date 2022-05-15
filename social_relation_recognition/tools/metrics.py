import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder


class CumulativeMeter(object):
    """Holds a sum of values along with their average, maximum ,minimum, step and epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the meter."""

        self.value = 0.
        self.sum = 0.
        self.count = 0.
        self.average = 0.

        self.step = 0
        self.batch = 0
        self.epoch = 0

        self.maximum = {
            'value': float('-inf'),
            'step': self.step,
            'epoch': self.epoch
        }

        self.minimum = {
            'value': float('inf'),
            'step': self.step,
            'epoch': self.epoch
        }

    def update(self, epoch, batch, step, value, count=1):
        """Update the meter with the given value and count."""

        self.value = value
        self.sum += value * count
        self.count += count
        self.average = self.sum / self.count

        self.step = step
        self.batch = batch
        self.epoch = epoch

        if (self.value > self.maximum['value']):
            self.maximum['value'] = self.value
            self.maximum['step'] = self.step
            self.maximum['batch'] = self.batch
            self.maximum['epoch'] = self.epoch

        if (self.value < self.minimum['value']):
            self.minimum['value'] = self.value
            self.minimum['step'] = self.step
            self.minimum['batch'] = self.batch
            self.minimum['epoch'] = self.epoch


def get_metrics(
    prediction_scores,
    labels,
    options=[
        'accuracy',
        'balanced_accuracy',
        'mean_average_precision',
        'class_recall',
        'class_precision',
        'class_average_precision',
        'log_loss'
    ]
):
    """Calculate multiple metrics for the given prediction scores and labels."""

    results = {}
    encoder = OneHotEncoder()

    # Convert nan values and get the total number of classes
    prediction_scores = np.nan_to_num(prediction_scores)
    number_classes = prediction_scores.shape[1]

    for option in options:

        if (option == 'accuracy'):
            scores = metrics.accuracy_score(
                labels,
                np.argmax(prediction_scores, axis=1)
            )

        elif (option == 'balanced_accuracy'):
            scores = metrics.recall_score(
                labels,
                np.argmax(prediction_scores, axis=1),
                labels=list(range(number_classes)),
                average='macro'
            )

        elif (option == 'mean_average_precision'):
            encoder.fit(np.arange(number_classes).reshape(-1, 1))
            labels_oh = encoder.transform(
                np.asarray(labels).reshape(-1, 1)).toarray()

            scores = metrics.average_precision_score(
                labels_oh,
                prediction_scores,
                average='macro'
            )

        elif (option == 'class_recall'):
            scores = metrics.recall_score(
                labels,
                np.argmax(prediction_scores, axis=1),
                labels=list(range(number_classes)),
                average=None
            )

        elif (option == 'class_precision'):
            scores = metrics.precision_score(
                labels,
                np.argmax(prediction_scores, axis=1),
                labels=list(range(number_classes)),
                average=None
            )

        elif (option == 'class_average_precision'):
            encoder.fit(np.arange(number_classes).reshape(-1, 1))
            labels_oh = encoder.transform(
                np.asarray(labels).reshape(-1, 1)).toarray()

            scores = metrics.average_precision_score(
                labels_oh,
                prediction_scores,
                average=None
            )

        elif (option == 'log_loss'):
            scores = metrics.log_loss(
                labels,
                prediction_scores,
                labels=list(range(number_classes))
            )

        else:
            continue

        results[option] = scores

    return results

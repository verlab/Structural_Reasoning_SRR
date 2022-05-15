"""
    Extract dataset statistics.
"""


import argparse
import os
import sys

import cv2
import numpy
from PIL import Image
from tqdm import tqdm

from test_features import analyze_processed_data
from tools.utils import check_python_3, load_json_file, save_json_file
from tools.utils_dataset import dataset_type2class, splits

CHANNEL_NUM = 3


def sum_values(dict_input, dict_output):

    for key, value in dict_input.items():

        if type(value) is dict:
            if key not in dict_output:
                dict_output[key] = {}

            sum_values(value, dict_output[key])

        else:
            if key not in dict_output:
                dict_output[key] = value
            else:
                dict_output[key] += value


def save_data(path, name, data, order=False):

    # Create folder
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save json
    path_save_file = os.path.join(path, name + '.json')
    save_json_file(path_save_file, data, order)


def show_data(data):

    for key in data:
        print('>>       [%s]' % key.capitalize())
        print(data[key])


def count_class_occurrences(metadata, type_relation, path_splits, path_save, show=False):
    """None."""

    # Initialize dict
    occurences = {}
    occurences['full'] = {}

    print('>>     Counting occurrences...')

    # Process splits data
    for split in splits:
        path_split = os.path.join(
            path_splits, args.type_relation, split + '.json')
        data_split = load_json_file(path_split)

        print('>>       Analysing %s metadata...' % split)
        occurences[split] = analyze_processed_data(
            metadata, data_split, args.type_relation)
        print('>>       ...done!')

        # Saving data
        path_save_split = os.path.join(path_save, type_relation, split)
        save_data(path_save_split, 'Occurrences', occurences[split])

    # Get full occurrences
    for split in splits:
        sum_values(occurences[split], occurences['full'])

    # Save full occurrences
    path_save_full = os.path.join(path_save, type_relation)
    save_data(path_save_full, 'Occurrences', occurences['full'])

    # Show data
    if show:
        show_data(occurences)

    print('>>     ... done!')

    return occurences


def calculate_class_weights(occurences, type_relation, number_classes, path_save, show=False):
    """(1/#samples_class) * (total/#classes)"""

    # Initialize dict
    weights = {}
    weights['full'] = {}

    print('>>     Calculating class weights...')

    # Calculate splits weights
    for key in occurences:

        # Initialize
        weights[key] = {}

        # total/#classes
        total_per_class = float(
            occurences[key]['relations_total']/number_classes)

        for label in range(number_classes):

            # 1/#samples_class
            inverse_per_class = float(1./occurences[key]['relations_class']
                                      [label]) if label in occurences[key]['relations_class'] else 0.
            weights[key][label] = inverse_per_class * total_per_class

        # Saving data
        path_save_folder = os.path.join(
            path_save, type_relation) if key == 'full' else os.path.join(path_save, type_relation, key)
        save_data(path_save_folder, 'Weights', weights[key])

    # Show data
    if show:
        show_data(weights)

    print('>>     ... done!')

    return weights


def calculate_normalization_scores(type_relation, path_splits, path_images_full, path_save, show=False):
    """Calculate dataset specific normalization scores."""

    # Initialize dict
    normalization = {}

    print('>>     Calculating normalization scores...')

    pixel_num_full = 0
    channel_sum_full = numpy.zeros(CHANNEL_NUM)
    channel_sum_squared_full = numpy.zeros(CHANNEL_NUM)

    # Process splits data
    for split in splits:

        # Initialize data
        path_split = os.path.join(
            path_splits, args.type_relation, split + '.json')
        data_split = load_json_file(path_split)

        pixel_num_split = 0
        channel_sum_split = numpy.zeros(CHANNEL_NUM)
        channel_sum_squared_split = numpy.zeros(CHANNEL_NUM)

        for id_image in tqdm(data_split):

            # Load image
            str_image = id_image.zfill(5) + '.jpg'
            path_image = os.path.join(path_images_full, str_image)
            full_image = numpy.array(Image.open(path_image).convert('RGB'))

            # Normalize and get num of pixels
            full_image = full_image/255.0
            pixel_num_split += (full_image.size/CHANNEL_NUM)

            # Get sum and sum squared
            channel_sum_split += numpy.sum(full_image, axis=(0, 1))
            channel_sum_squared_split += numpy.sum(
                numpy.square(full_image), axis=(0, 1))

        # Get split values
        mean_split = channel_sum_split / pixel_num_split
        std_split = numpy.sqrt(
            (channel_sum_squared_split / pixel_num_split) - numpy.square(mean_split))

        normalization[split] = {
            'mean': mean_split.tolist(), 'std': std_split.tolist()}

        # Get full values
        pixel_num_full += pixel_num_split
        channel_sum_full += channel_sum_split
        channel_sum_squared_full += channel_sum_squared_split

        # Saving data
        path_save_split = os.path.join(path_save, type_relation, split)
        save_data(path_save_split, 'Normalization', normalization[split])

    # Get full values
    mean_full = channel_sum_full / pixel_num_full
    std_full = numpy.sqrt(
        (channel_sum_squared_full / pixel_num_full) - numpy.square(mean_full))

    normalization['full'] = {
        'mean': mean_full.tolist(), 'std': std_full.tolist()}

    # Saving data
    path_save_full = os.path.join(path_save, type_relation)
    save_data(path_save_full, 'Normalization', normalization['full'])

    # Show data
    if show:
        show_data(normalization)

    print('>>     ... done!')


def extract_statistics(args):
    """None."""

    # Check for python correct version
    check_python_3()

    # Getting total number of classes
    number_classes = dataset_type2class[args.dataset][args.type_relation]

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_save = os.path.join(path_dataset, 'statistics')
    path_metadata = os.path.join(path_dataset, args.dataset + '.json')
    path_splits = os.path.join(path_dataset, 'splits')
    path_images_full = os.path.join(path_dataset, 'images', 'full')

    # Load metadata
    metadata = load_json_file(path_metadata)

    print('>> [EXTRACTING STATISTICS]')
    print('>>   %s %s data...' % (args.dataset, args.type_relation))
    print('>>   Saving statistics to: %s' % (path_save))

    # Count occurrences
    occurrences = count_class_occurrences(
        metadata, args.type_relation, path_splits, path_save, args.show)

    # Get weights
    weights = calculate_class_weights(
        occurrences, args.type_relation, number_classes, path_save, args.show)

    # Get normalization
    normalization = calculate_normalization_scores(
        args.type_relation, path_splits, path_images_full, path_save, args.show)

    print('>>   ... done!')
    print('>>   %s %s statistics extraction finished!' %
          (args.dataset, args.type_relation))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Statistcs extracting')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])
    parser.add_argument('type_relation',  type=str, help='type of data', choices=[
        'domain',
        'relationship'
    ])
    parser.add_argument('--show', action='store_true', help='show images')
    parser.add_argument('--debug', action='store_true',
                        help='activate debug mode')

    args = parser.parse_args()
    extract_statistics(args)

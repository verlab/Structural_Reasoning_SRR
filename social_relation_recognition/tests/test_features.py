"""
    Tests for metadata and extracted features.
"""


import argparse
import os
import sys

import h5py

from tools.utils import (check_python_3, load_hdf5_dataset, load_json_file,
                         save_string_list_file)
from tools.utils_dataset import (dataset_type_split2relations, feature2size,
                                 feature_types, relations, splits)


def analyze_processed_data(data_processed, list_ids, type):
    """Count total, unique and per class numbers of relations, persons and objects."""

    # Initialize counter
    counter = {
        'images_total': 0,
        'persons_total': 0,
        'persons_unique': 0,
        'relations_total': 0,
        'relations_class': {},
        'objects_total': 0,
        'objects_class': {}
    }

    # Progression counter
    fraction = len(list_ids) * 0.1
    porcent = 0

    for index, id in enumerate(list_ids):

        # Progression counter
        if (int(index % fraction) == 0):
            print('>>       ...{0:.0%}'.format(porcent))
            porcent += 0.1

        # Count total images
        counter['images_total'] += 1

        # Count total number of persons
        counter['persons_total'] += len(data_processed[id]['body_bbox'])

        # Initialize unique person counter and get relations
        counter_relation_persons = []
        relations = data_processed[id][type]

        # Get relation info
        for relation in relations:

            # Count relation total
            counter['relations_total'] += 1

            # Get persons
            split_relation = relation.split()

            # Count unique persons per relation
            for person in split_relation:
                if (person not in counter_relation_persons):
                    counter_relation_persons.append(person)

            # Count relation per class
            label = relations[relation]

            if (label in counter['relations_class']):
                counter['relations_class'][label] += 1

            else:
                counter['relations_class'][label] = 1

        # Add unique persons count to total
        counter['persons_unique'] += len(counter_relation_persons)

        # Count total number of objects
        counter['objects_total'] += len(data_processed[id]['object']['bbox'])

        # Get object class info
        categories = data_processed[id]['object']['category']

        # Count objects per class
        for category in categories:

            if (str(category) in counter['objects_class']):
                counter['objects_class'][str(category)] += 1

            else:
                counter['objects_class'][str(category)] = 1

    print('>>       ...{0:.0%}'.format(porcent))

    return counter


def analyze_hdf5_relations(path_file):
    """Count total, unique, per size, per data type and per class numbers of images, relations and persons."""

    # Initialize counter
    counter = {
        'images_total': 0,
        'persons_unique': 0,
        'relations_total': 0,
        'features_size': {},
        'features_dtype': {},
        'relations_class': {}
    }

    with h5py.File(path_file, 'r') as file:

        # Progression counter
        fraction = len(file.keys()) * 0.1
        porcent = 0

        for index, id_image in enumerate(file.keys()):

            # Progression counter
            if (int(index % fraction) == 0):
                print('>>       ...{0:.0%}'.format(porcent))
                porcent += 0.1

            # Count total images
            counter['images_total'] += 1

            # Initialize unique person counter
            counter_relation_persons = []

            # Get image
            data_image = file[id_image]

            # Get persons in the image
            for id_relation in data_image.keys():

                # Count relation total
                counter['relations_total'] += 1

                # Get persons
                split_relation = id_relation.split()

                # Count unique persons per relation
                for person in split_relation:
                    if (person not in counter_relation_persons):
                        counter_relation_persons.append(person)

                # Get feature size, data type and class label
                data_relation = data_image[id_relation]
                size_feature = data_relation.size
                type_data = data_relation.dtype
                label_class = data_relation.attrs['label']

                # Count persons per feature size
                if (size_feature in counter['features_size']):
                    counter['features_size'] += 1

                else:
                    counter['features_size'] = 1

                # Count persons per data type
                if (type_data in counter['features_dtype']):
                    counter['features_dtype'] += 1

                else:
                    counter['features_dtype'] = 1

                # Count relations per class
                if (label_class in counter['relations_class']):
                    counter['relations_class'] += 1

                else:
                    counter['relations_class'] = 1

            # Add unique persons count to total
            counter['persons_unique'] += len(counter_relation_persons)

    print('>>       ...{0:.0%}'.format(porcent))

    return counter


def analyze_hdf5_persons(path_file):
    """Count total, per size and per data type numbers of images and persons."""

    # Initialize counter
    counter = {
        'images_total': 0,
        'persons_total': 0,
        'features_size': {},
        'features_dtype': {}
    }

    with h5py.File(path_file, 'r') as file:

        # Progression counter
        fraction = len(file.keys()) * 0.1
        porcent = 0

        for index, id_image in enumerate(file.keys()):

            # Progression counter
            if (int(index % fraction) == 0):
                print('>>       ...{0:.0%}'.format(porcent))
                porcent += 0.1

            # Count total images
            counter['images_total'] += 1

            # Get image
            data_image = file[id_image]

            # Get persons in the image
            for id_person in data_image.keys():

                # Count total persons
                counter['persons_total'] += 1

                # Get feature size and data type
                data_person = data_image[id_person]
                size_feature = data_person.size
                type_data = data_person.dtype

                # Count persons per feature size
                if (size_feature in counter['features_size']):
                    counter['features_size'] += 1

                else:
                    counter['features_size'] = 1

                # Count persons per data type
                if (type_data in counter['features_dtype']):
                    counter['features_dtype'] += 1

                else:
                    counter['features_dtype'] = 1

    print('>>       ...{0:.0%}'.format(porcent))

    return counter


def analyze_hdf5_objects(path_file):
    """Count total, per size, per data type and per class numbers of images and objects."""

    # Initialize counter
    counter = {
        'images_total': 0,
        'objects_total': 0,
        'features_size': {},
        'features_dtype': {},
        'objects_class': {}
    }

    with h5py.File(path_file, 'r') as file:

        # Progression counter
        fraction = len(file.keys()) * 0.1
        porcent = 0

        for index, id_image in enumerate(file.keys()):

            # Progression counter
            if (int(index % fraction) == 0):
                print('>>       ...{0:.0%}'.format(porcent))
                porcent += 0.1

            # Count total images
            counter['images_total'] += 1

            # Get image
            data_image = file[id_image]

            # Get objects in the image
            for id_object in data_image.keys():

                # Count total objects
                counter['objects_total'] += 1

                # Get feature size and data type
                data_object = data_image[id_object]
                size_feature = data_object.size
                type_data = data_object.dtype
                label_class = data_object.attrs['label']

                # Count objects per feature size
                if (size_feature in counter['features_size']):
                    counter['features_size'] += 1

                else:
                    counter['features_size'] = 1

                # Count objects per data type
                if (type_data in counter['features_dtype']):
                    counter['features_dtype'] += 1

                else:
                    counter['features_dtype'] = 1

                # Count objects per class
                if (label_class in counter['objects_class']):
                    counter['objects_class'] += 1

                else:
                    counter['objects_class'] = 1

    print('>>       ...{0:.0%}'.format(porcent))

    return counter


def test_global_metadata(args):
    """Run global features analysis for features and metada then compare both"""

    # Check for python correct version
    check_python_3()

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_splits = os.path.join(path_dataset, 'splits')
    path_hdf5_features = os.path.join(path_dataset, 'features', 'hdf5')
    path_processed_data = os.path.join(path_dataset, args.dataset + '.json')

    # Loading data
    data_processed = load_json_file(path_processed_data)

    print('>> [TESTING]')

    # Dictionary to hold the analisys data
    dictionary_analisys = {}

    # Build analysis dictionary
    for relation in relations:

        print('>>   Processing %s %s global data...' %
              (args.dataset, relation))

        dictionary_analisys[relation] = {}

        # Get path and add dict
        path_split_type = os.path.join(path_splits, relation)

        for split in splits:

            dictionary_analisys[relation][split] = {}

            # Loading split data
            path_split = os.path.join(path_split_type, split)
            data_split = load_json_file(path_split)

            print('>>     Analyzing processed metadata...')

            # Analyze processed metadata
            dictionary_analisys[relation][split]['processed_data'] = analyze_processed_data(
                data_processed, data_split, relation)
            dictionary_analisys[relation][split]['features'] = {}

            print('>>     ...done!')

            # Analyze feature metada per type
            for feature in feature_types:

                print('>>     Analyzing %s %s features...' % (split, feature))

                path_feature = os.path.join(
                    path_hdf5_features, relation, split, feature, '.hdf5')
                split_feature = feature.split('_')

                if ((split_feature[0] == 'context') or (split_feature[0] == 'first')):
                    dictionary_analisys[relation][split]['features'][feature] = analyze_hdf5_relations(
                        path_feature)

                elif (split_feature[0] == 'body'):
                    dictionary_analisys[relation][split]['features'][feature] = analyze_hdf5_persons(
                        path_feature)

                elif ((split_feature[0] == 'object') or (split_feature[0] == 'first')):
                    dictionary_analisys[relation][split]['features'][feature] = analyze_hdf5_objects(
                        path_feature)

                else:
                    print('>> [ERROR] Undefined feature type')
                    sys.exit(1)

                print('>>     ...done!')

        print('>>   ...done!')

    # Compare dictionary data
    for relation in dictionary_analisys:

        print('>>   Testing %s %s global data...' % (args.dataset, relation))

        for split in dictionary_analisys[relation]:

            # Compare per feature metada with processed metadata
            for feature in dictionary_analisys[relation][split]['features']:

                print('>>     Comparing %s %s features...' % (split, feature))

                generated_size = dictionary_analisys[relation][split]['features'][feature]['features_size']
                ground_truth_size = feature2size[feature]

                assert ground_truth_size == generated_size, \
                    '>> [ERROR] Feature size missmatch for %s features' % feature

                generated_dtype = dictionary_analisys[relation][split]['features'][feature]['features_dtype']
                ground_truth_type = 'float32'

                assert ground_truth_type == generated_dtype, \
                    '>> [ERROR] Feature type missmatch for %s features' % feature

                if ('images_total' in dictionary_analisys[relation][split]['features'][feature]):

                    ground_truth = dataset_type_split2relations[args.dataset][relation][split]
                    generated = dictionary_analisys[relation][split]['features'][feature]['images_total']

                    assert ground_truth == generated, \
                        '>> [ERROR] Total number of images missmatch for %s features' % feature

                if ('persons_total' in dictionary_analisys[relation][split]['features'][feature]):

                    ground_truth = dictionary_analisys[relation][split]['processed_data']['persons_total']
                    generated = dictionary_analisys[relation][split]['features'][feature]['persons_total']

                    assert ground_truth == generated, \
                        '>> [ERROR] Total number of persons missmatch for %s features' % feature

                if ('persons_unique' in dictionary_analisys[relation][split]['features'][feature]):

                    ground_truth = dictionary_analisys[relation][split]['processed_data']['persons_unique']
                    generated = dictionary_analisys[relation][split]['features'][feature]['persons_unique']

                    assert ground_truth == generated, \
                        '>> [ERROR] Total number of unique persons missmatch for %s features' % feature

                if ('relations_total' in dictionary_analisys[relation][split]['features'][feature]):

                    ground_truth = dictionary_analisys[relation][split]['processed_data']['relations_total']
                    generated = dictionary_analisys[relation][split]['features'][feature]['relations_total']

                    assert ground_truth == generated, \
                        '>> [ERROR] Total number of relations missmatch for %s features' % feature

                if ('objects_total' in dictionary_analisys[relation][split]['features'][feature]):

                    ground_truth = dictionary_analisys[relation][split]['processed_data']['objects_total']
                    generated = dictionary_analisys[relation][split]['features'][feature]['objects_total']

                    assert ground_truth == generated, \
                        '>> [ERROR] Total number of objects missmatch for %s features' % feature

                print('>>     ...done!')

        print('>>   ...done!')

    print('>>   %s global metada test finished!' % args.dataset)


def local_metadata_test(args):
    """Run through metadata for each split and compare if the keys exist inside hdmf5 for persons, objects and relations"""

    # Check for python correct version
    check_python_3()

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_splits = os.path.join(path_dataset, 'splits')
    path_hdf5_features = os.path.join(path_dataset, 'features', 'hdf5')
    path_processed_data = os.path.join(path_dataset, args.dataset + '.json')

    # Loading data
    data_processed = load_json_file(path_processed_data)

    print('>> [TESTING]')

    # Build dictionary
    for relation in relations:

        # Get path and add dict
        path_split_type = os.path.join(path_splits, relation)

        for split in splits:

            print('>>   Testing %s %s %s local data...' %
                  (args.dataset, relation, split))

            # Loading split data
            path_split = os.path.join(path_split_type, split)
            data_split = load_json_file(path_split)

            data_features = {}

            # Loading hdf5 data
            for feature in feature_types:

                print('>>     Loading %s features...' % (feature))

                path_feature = os.path.join(
                    path_hdf5_features, relation, split, feature, '.hdf5')
                data_features[feature] = load_hdf5_dataset(path_feature)

            # Progression counter
            fraction = len(data_split) * 0.1
            porcent = 0

            for index, id_image in enumerate(data_split):

                # Progression counter
                if (int(index % fraction) == 0):
                    print('>>     ...{0:.0%}'.format(porcent))
                    porcent += 0.1

                    for id_person, body_box in enumerate(data_split[id]['body_box']):

                        for feature in data_features:

                            split_feature = feature.split('_')

                            if (split_feature[0] == 'body'):

                                assert str(id_person + 1) in data_features[feature]['data'][id_image]['data'], \
                                    '>> [ERROR] Missing person key for %s features' % feature

                    for relation in enumerate(data_split[id]['relationship']):

                        for feature in data_features:

                            if ((split_feature[0] == 'context') or (split_feature[0] == 'first')):

                                assert relation in data_features[feature]['data'][id_image]['data'], \
                                    '>> [ERROR] Missing person key for %s features' % feature

            print('>>     ...{0:.0%}'.format(porcent))

        print('>>   ...done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Features testing')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])

    args = parser.parse_args()
    test_global_metadata(args)

"""
    Graph related routines and configurations.
"""


import os

import dgl
import numpy
import torch

from tools.utils import load_hdf5_dataset
from tools.utils_dataset import calculate_relations, scale2attributes


def load_attributes(scales, path_attributes):

    dataset_attributes = {}

    for scale in scales:

        dataset_attributes[scale] = {}
        for attribute in scale2attributes[scale]:

            path_data_features = os.path.join(
                path_attributes, attribute + '.hdf5')
            print('>>     - %s: %s' % (attribute, path_data_features))
            data_features = load_hdf5_dataset(path_data_features)
            dataset_attributes[scale][attribute] = data_features

    return dataset_attributes


def load_features(scales, path_features, type, attributes=False):

    dataset_features = {}

    path_data_features = os.path.join(
        path_features, 'social_scale_' + type + '.hdf5')
    print('>>     - %s: %s' % ('social scale', path_data_features))
    data_features = load_hdf5_dataset(path_data_features)
    dataset_features['social_scale'] = data_features

    if attributes:
        for scale in scales:

            dataset_features[scale] = {}
            for attribute in scale2attributes[scale]:

                path_data_features = os.path.join(
                    path_features, attribute + '.hdf5')
                print('>>     - %s: %s' % (attribute, path_data_features))
                data_features = load_hdf5_dataset(path_data_features)
                dataset_features[scale][attribute] = data_features

    return dataset_features


def get_attributes(scales):

    attributes = []

    for scale in scales:
        for attribute in scale2attributes[scale]:

            attributes.append(attribute)

    return attributes


def set_graph(scales, topology, relation_edges, use_loops=True):

    # Node and edge types
    type_nodes = ['relation']
    type_edges = []

    if (relation_edges != 'none'):
        type_edges.append('relation')

    if use_loops:
        type_edges.append('self')

    for scale in scales:

        type_nodes.append(scale)
        type_edges.append(scale)

        if (topology != 'SKG'):
            for attribute in scale2attributes[scale]:

                type_nodes.append(attribute)
                type_edges.append(attribute)

    return type_nodes, type_edges


def load_graph(scales, type_relation, data_meta, data_ids, data_features, use_residual=False, use_attributes=False):

    # Edges, features and labels
    graph_features = {}
    graph_edges = {}
    graph_labels = []

    # Store index for nodes and targets
    indexes_personal = {}
    indexes_local = {}
    indexes_global = {}
    indexes_relation = {}

    # Counters
    counter_personal = 0
    counter_local = 0
    counter_global = 0
    counter_objects = 0
    counter_relation = 0

    # Neighors Nodes
    neighbors = {}

    # Initialize node features lists
    for scale in scales:
        graph_features[scale] = []

        if use_attributes:
            for attribute in scale2attributes[scale]:
                graph_features[attribute] = []

    # Initialize edges lists
    for scale in scales:

        triplet_scale = (scale, scale, 'relation')
        sources_scale = []
        targets_scale = []

        graph_edges[triplet_scale] = (sources_scale, targets_scale)

        if use_attributes:
            for attribute in scale2attributes[scale]:

                triplet_attribute = (attribute, attribute, scale)
                sources_attribute = []
                targets_attribute = []

                graph_edges[triplet_attribute] = (
                    sources_attribute, targets_attribute)

    # Initialize relation nodes and edges
    graph_features['relation'] = []

    triplet_relation = ('relation', 'relation', 'relation')
    sources_relation = []
    targets_relation = []

    triplet_self = ('relation', 'self', 'relation')
    sources_self = []
    targets_self = []

    graph_edges[triplet_relation] = (sources_relation, targets_relation)
    graph_edges[triplet_self] = (sources_self, targets_self)

    # Add scales data
    for id_image in data_ids:

        image_scales_data = data_features['social_scale']['data'][id_image]['data']
        image_relation_data = data_meta[id_image][type_relation]

        for id_data in image_scales_data:

            # personal_N, local_N M, global
            id_split = id_data.split('_')
            prefix = id_split[0]
            sufix = id_split[-1]

            if ((prefix == 'personal') and ('personal' in scales)):

                # Key for identifying the node globaly
                key_personal = id_image + '_' + sufix

                assert not(key_personal in indexes_personal), \
                    '>> [ERROR] Redundant personal context key: %s' % key_personal

                indexes_personal[key_personal] = counter_personal
                neighbors[key_personal] = []

                # Add personal features
                features_scale = image_scales_data[id_data]['data']
                graph_features['personal'].append(features_scale)

                # Add attribute features and edges
                if use_attributes:
                    for attribute in scale2attributes['personal']:

                        features_attribute = data_features['personal'][
                            attribute]['data'][id_image]['data'][sufix]['data']
                        graph_features[attribute].append(features_attribute)

                        triplet = (attribute, attribute, 'personal')
                        graph_edges[triplet][0].append(counter_personal)
                        graph_edges[triplet][1].append(counter_personal)

                counter_personal += 1

            if ((prefix == 'local') and ('local' in scales)):

                # Key for identifying the node globaly
                key_local = id_image + '_' + sufix

                assert not(key_local in indexes_local), \
                    '>> [ERROR] Redundant local context key: %s' % key_local

                indexes_local[key_local] = counter_local

                # Add local features
                features_scale = image_scales_data[id_data]['data']
                graph_features['local'].append(features_scale)

                # Add attribute features and edges
                if use_attributes:
                    for attribute in scale2attributes['local']:

                        features_attribute = data_features['local'][attribute]['data'][id_image]['data'][sufix]['data']
                        graph_features[attribute].append(features_attribute)

                        triplet = (attribute, attribute, 'local')
                        graph_edges[triplet][0].append(counter_local)
                        graph_edges[triplet][1].append(counter_local)

                counter_local += 1

            if ((prefix == 'global') and ('global' in scales)):

                # Key for identifying the node globaly
                key_global = id_image + '_' + sufix

                assert not(key_global in indexes_global), \
                    '>> [ERROR] Redundant global context key: %s' % key_global

                indexes_global[key_global] = counter_global

                # Add global features
                features_scale = image_scales_data[id_data]['data']
                graph_features['global'].append(features_scale)

                # Add attribute features and edges
                if use_attributes:
                    for attribute in scale2attributes['global']:

                        # Get all objects features
                        features_attribute = data_features['global'][attribute]['data'][id_image]['data']

                        for feature in features_attribute:

                            graph_features[attribute].append(
                                features_attribute[feature]['data'])

                            triplet = (attribute, attribute, 'global')
                            graph_edges[triplet][0].append(counter_objects)
                            graph_edges[triplet][1].append(counter_global)

                            counter_objects += 1

                counter_global += 1

        for relation in image_relation_data:

            # Add relation key
            indexes_relation[relation] = counter_relation

            # Use final or blank relation features
            if use_residual:
                # Relation key
                key_relation = 'final_' + relation

                graph_features['relation'].append(
                    image_scales_data[key_relation]['data'])

            else:
                graph_features['relation'].append(
                    numpy.zeros((2048), dtype=numpy.float32))

            # Add self edges
            graph_edges[triplet_self][0].append(counter_relation)
            graph_edges[triplet_self][1].append(counter_relation)

            label = image_relation_data[relation]
            graph_labels.append(label)

            if ('personal' in scales):

                # Get personal keys
                split_space = relation.split()
                key_personal_1 = id_image + '_' + split_space[0]
                key_personal_2 = id_image + '_' + split_space[1]

                triplet = ('personal', 'personal', 'relation')

                graph_edges[triplet][0].append(
                    indexes_personal[key_personal_1])
                graph_edges[triplet][1].append(counter_relation)

                graph_edges[triplet][0].append(
                    indexes_personal[key_personal_2])
                graph_edges[triplet][1].append(counter_relation)

                # Add neighbors
                neighbors[key_personal_1].append(relation)
                neighbors[key_personal_2].append(relation)

            if ('local' in scales):

                # Get local key
                key_local = id_image + '_' + relation

                triplet = ('local', 'local', 'relation')

                graph_edges[triplet][0].append(indexes_local[key_local])
                graph_edges[triplet][1].append(counter_relation)

            if ('global' in scales):

                # Get global key
                key_global = id_image + '_global'

                triplet = ('global', 'global', 'relation')

                graph_edges[triplet][0].append(indexes_global[key_global])
                graph_edges[triplet][1].append(counter_relation)

            counter_relation += 1

    # Add relation edges
    for personal in neighbors:
        for relation_1 in neighbors[personal]:
            for relation_2 in neighbors[personal]:

                if (relation_1 != relation_2):

                    graph_edges[triplet_relation][0].append(
                        indexes_relation[relation_1])
                    graph_edges[triplet_relation][1].append(
                        indexes_relation[relation_2])

    # Get graph
    graph = dgl.heterograph(graph_edges)

    # Test and convert to tensor
    assert counter_relation == len(graph_labels), \
        '>> [ERROR] Number of relations missmatch with number of labels: \n>>   Relations: %d\n>>   Labels: %d' % (
            counter_relation, len(graph_labels))

    graph_labels = torch.tensor(graph_labels)

    for features in graph_features:

        size_features = len(graph_features[features])
        size_counter = 0

        if ((features == 'personal') or (features in scale2attributes['personal'])):
            size_counter = counter_personal

        elif ((features == 'local') or (features in scale2attributes['local'])):
            size_counter = counter_local

        elif (features == 'global'):
            size_counter = counter_global

        elif (features in scale2attributes['global']):
            size_counter = counter_objects

        else:
            size_counter = counter_relation

        assert size_features == size_counter, \
            '>> [ERROR] List features size missmatch for %s features: \n>>   Features: %d\n>>   Counter: %d' % (
                features, size_features, size_counter)

        graph_features[features] = torch.tensor(
            graph_features[features]).view(size_features, -1)

    return graph, graph_labels, graph_features


def build_graph(
    scales,
    topology,
    relation_edges,
    data_meta,
    data_relation,
    data_id,
    data_attributes=None,
):

    # Node indexes, edges, features and labels
    graph_indexes = {}
    graph_edges = {}
    graph_features = {}

    # Node counters
    counter_relation = 0
    counter_personal = 0
    counter_local = 0
    counter_global = 0
    counter_objects = 0

    # Initialize indexes
    graph_indexes['relation'] = {}

    for scale in scales:
        graph_indexes[scale] = {}

    # Social neighors list
    if (relation_edges == 'neighbors'):
        social_neighbors = {}

    # Check attributes data and initialize nodes lists
    if (topology != 'SKG'):

        assert data_attributes, \
            '>> [ERROR] Missing attributes data'

        for scale in scales:
            for attribute in scale2attributes[scale]:
                graph_features[attribute] = []

    # Initialize edges lists
    for scale in scales:

        triplet_scale = (scale, scale, 'relation')
        sources_scale = []
        targets_scale = []

        graph_edges[triplet_scale] = (sources_scale, targets_scale)

        if (topology != 'SKG'):
            for attribute in scale2attributes[scale]:

                triplet_attribute = (attribute, attribute, scale)
                sources_attribute = []
                targets_attribute = []

                graph_edges[triplet_attribute] = (
                    sources_attribute, targets_attribute)

    # Initialize relation nodes list
    graph_features['relation'] = []

    # Initialize relation edges list
    if (relation_edges != 'none'):

        triplet_relation = ('relation', 'relation', 'relation')
        sources_relation = []
        targets_relation = []

        graph_edges[triplet_relation] = (sources_relation, targets_relation)

    # Add scales edges
    for id_image in data_id:

        image_metadata = data_meta[id_image]

        for relation in image_metadata[data_relation]:

            split_space = relation.split()
            id_person_1 = split_space[0]
            id_person_2 = split_space[1]

            # Get unique node keys
            key_relation = id_image + '_' + relation
            key_personal_1 = id_image + '_' + id_person_1
            key_personal_2 = id_image + '_' + id_person_2
            key_local = id_image + '_' + relation
            key_global = id_image

            # Count unique relationships
            graph_indexes['relation'][key_relation] = counter_relation

            # Add blank relation features
            graph_features['relation'].append(
                numpy.zeros((2048), dtype=numpy.float32))

            if ('personal' in scales):

                # Count unique individuals
                if not(key_personal_1 in graph_indexes['personal']):

                    graph_indexes['personal'][key_personal_1] = counter_personal

                    # Initialize neighbors list
                    if (relation_edges == 'neighbors'):
                        social_neighbors[key_personal_1] = []

                    if (topology != 'SKG'):
                        for attribute in scale2attributes['personal']:

                            # Add attribute features
                            features_attribute = data_attributes['personal'][attribute][
                                'data'][id_image]['data'][id_person_1]['data']
                            graph_features[attribute].append(
                                features_attribute)

                            # Add attribute edges
                            triplet = (attribute, attribute, 'personal')

                            graph_edges[triplet][0].append(counter_personal)
                            graph_edges[triplet][1].append(counter_personal)

                    counter_personal += 1

                if not(key_personal_2 in graph_indexes['personal']):

                    graph_indexes['personal'][key_personal_2] = counter_personal

                    # Initialize neighbors list
                    if (relation_edges == 'neighbors'):
                        social_neighbors[key_personal_2] = []

                    if (topology != 'SKG'):
                        for attribute in scale2attributes['personal']:

                            # Add attribute features
                            features_attribute = data_attributes['personal'][attribute][
                                'data'][id_image]['data'][id_person_2]['data']
                            graph_features[attribute].append(
                                features_attribute)

                            # Add attribute edges
                            triplet = (attribute, attribute, 'personal')

                            graph_edges[triplet][0].append(counter_personal)
                            graph_edges[triplet][1].append(counter_personal)

                    counter_personal += 1

                # Add personal scale edges
                triplet = ('personal', 'personal', 'relation')

                graph_edges[triplet][0].append(
                    graph_indexes['personal'][key_personal_1])
                graph_edges[triplet][1].append(
                    graph_indexes['relation'][key_relation])

                graph_edges[triplet][0].append(
                    graph_indexes['personal'][key_personal_2])
                graph_edges[triplet][1].append(
                    graph_indexes['relation'][key_relation])

            if ('local' in scales):

                graph_indexes['local'][key_local] = counter_local

                if (topology != 'SKG'):
                    for attribute in scale2attributes['local']:

                        # Add attribute features
                        features_attribute = data_attributes['local'][attribute][
                            'data'][id_image]['data'][relation]['data']
                        graph_features[attribute].append(features_attribute)

                        # Add attribute edges
                        triplet = (attribute, attribute, 'local')
                        graph_edges[triplet][0].append(counter_local)
                        graph_edges[triplet][1].append(counter_local)

                counter_local += 1

                # Add local scale edges
                triplet = ('local', 'local', 'relation')

                graph_edges[triplet][0].append(
                    graph_indexes['local'][key_local])
                graph_edges[triplet][1].append(
                    graph_indexes['relation'][key_relation])

            if ('global' in scales):

                # Check key
                if not(key_global in graph_indexes['global']):
                    graph_indexes['global'][key_global] = counter_global

                    if (topology != 'SKG'):
                        for attribute in scale2attributes['global']:

                            # Get all objects features
                            features_attribute = data_attributes['global'][attribute]['data'][id_image]['data']

                            # More than one of the same attribute per global node
                            for feature in features_attribute:

                                # Generate unique key
                                key_attribute = key_global + '_' + attribute + '_' + feature

                                # Add attribute index
                                graph_indexes['global'][key_attribute] = counter_objects

                                # Add attribute features
                                graph_features[attribute].append(
                                    features_attribute[feature]['data'])

                                # Add attribute edges
                                triplet = (attribute, attribute, 'global')
                                graph_edges[triplet][0].append(counter_objects)
                                graph_edges[triplet][1].append(counter_global)

                                counter_objects += 1

                    counter_global += 1

                # Add global scale edges
                triplet = ('global', 'global', 'relation')

                graph_edges[triplet][0].append(
                    graph_indexes['global'][key_global])
                graph_edges[triplet][1].append(
                    graph_indexes['relation'][key_relation])

            # Add social neighbors
            if (relation_edges == 'neighbors'):
                social_neighbors[key_personal_1].append(relation)
                social_neighbors[key_personal_2].append(relation)

            counter_relation += 1

    # Add blank social scale features
    if (topology == 'SKG-'):
        if ('personal' in scales):
            graph_features['personal'] = numpy.zeros(
                (counter_personal, 2048), dtype=numpy.float32)

        if ('local' in scales):
            graph_features['local'] = numpy.zeros(
                (counter_local, 2048), dtype=numpy.float32)

        if ('global' in scales):
            graph_features['global'] = numpy.zeros(
                (counter_global, 2048), dtype=numpy.float32)

    # Add relation edges
    if (relation_edges == 'neighbors'):

        # Per person using social neighbors
        for personal in social_neighbors:

            split_underline = personal.split('_')

            id_image = split_underline[0]

            if (len(split_underline) > 2):
                id_image = id_image + '_' + split_underline[1]

            for relation_1 in social_neighbors[personal]:
                for relation_2 in social_neighbors[personal]:
                    if (relation_1 != relation_2):

                        key_relation_1 = id_image + '_' + relation_1
                        key_relation_2 = id_image + '_' + relation_2

                        graph_edges[triplet_relation][0].append(
                            graph_indexes['relation'][key_relation_1])
                        graph_edges[triplet_relation][1].append(
                            graph_indexes['relation'][key_relation_2])

    elif (relation_edges == 'full'):

        # Per image using all relations
        for id_image in data_id:

            image_metadata = data_meta[id_image]

            for relation_1 in image_metadata[data_relation]:
                for relation_2 in image_metadata[data_relation]:
                    if (relation_1 != relation_2):

                        key_relation_1 = id_image + '_' + relation_1
                        key_relation_2 = id_image + '_' + relation_2

                        graph_edges[triplet_relation][0].append(
                            graph_indexes['relation'][key_relation_1])
                        graph_edges[triplet_relation][1].append(
                            graph_indexes['relation'][key_relation_2])

    # Test features and convert to tensor
    for features in graph_features:

        size_features = len(graph_features[features])
        size_counter = 0

        if ((features == 'personal') or (features in scale2attributes['personal'])):
            size_counter = counter_personal

        elif ((features == 'local') or (features in scale2attributes['local'])):
            size_counter = counter_local

        elif (features == 'global'):
            size_counter = counter_global

        elif (features in scale2attributes['global']):
            size_counter = counter_objects

        else:
            size_counter = counter_relation

        assert size_features == size_counter, \
            '>> [ERROR] List size missmatch for %s features: \n>>   Features: %d\n>>   Counter: %d' % (
                features, size_features, size_counter)

        graph_features[features] = torch.tensor(
            graph_features[features]).view(size_features, -1)

    # Generate the graph
    graph = dgl.heterograph(graph_edges)

    return graph, graph_features

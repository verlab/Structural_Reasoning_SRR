# coding=utf-8

"""
    Extract and save attribute features with pre-trained models.
"""


import argparse
import os
import sys
import time
from datetime import date

import cv2
import h5py
import numpy as np

from tools.utils import (check_python_2, load_json_file, set_environment_caffe,
                         set_environment_hdf5)
from tools.utils_dataset import (crop_bounding_box, fix_bounding_box,
                                 get_bounding_boxes, get_context_bounding_box,
                                 splits)

"""Feature type to extraction model convertion."""
feature2model = {
    'object_attention': 'SENet.caffemodel',
    'context_emotion': 'group_scene.caffemodel',
    'context_activity': 'SituationCrf.caffemodel.h5',
    'body_activity': 'SituationCrf_body.caffemodel.h5',
    'body_age': 'body_age.caffemodel',
    'body_clothing': 'body_clothing.caffemodel',
    'body_gender': 'body_gender.caffemodel'
}

"""Model to Caffe protocol convertion."""
model2protocol = {
    'SENet.caffemodel': 'SENet.prototxt',
    'group_scene.caffemodel': 'group_scene.prototxt',
    'SituationCrf.caffemodel.h5': 'SituationCrf.prototxt',
    'SituationCrf_body.caffemodel.h5': 'SituationCrf.prototxt',
    'body_age.caffemodel': 'double_stream.prototxt',
    'body_clothing.caffemodel': 'double_stream.prototxt',
    'body_gender.caffemodel': 'double_stream.prototxt'
}

"""Model to extraction layer convertion."""
model2layer = {
    'SENet.caffemodel': 'pool5/7x7_s1',
    'group_scene.caffemodel': 'global_pool',
    'SituationCrf.caffemodel.h5': 'fc7',
    'SituationCrf_body.caffemodel.h5': 'fc7',
    'body_age.caffemodel': 'fc7',
    'body_clothing.caffemodel': 'fc7',
    'body_gender.caffemodel': 'fc7'
}

"""Model to Caffe library convertion."""
model2library = {
    'SENet.caffemodel': 'Caffe_SENet',
    'group_scene.caffemodel': 'Caffe_default',
    'SituationCrf.caffemodel.h5': 'Caffe_SituationCrf',
    'SituationCrf_body.caffemodel.h5': 'Caffe_SituationCrf',
    'body_age.caffemodel': 'Caffe_default',
    'body_clothing.caffemodel': 'Caffe_default',
    'body_gender.caffemodel': 'Caffe_default'
}


def caffe_import(path):
    """Import Caffe python libraries from the given path."""

    sys.path.insert(0, path)

    import caffe

    return caffe


def initialize_transformer(caffe, protocol):
    """Initialize the correct transformer shape and mean image for the given protocol."""

    if (protocol == 'double_stream.prototxt'):
        shape = (1, 3, 227, 227)
        transformer = caffe.io.Transformer({'data': shape})
        channel_mean = np.zeros((3, 227, 227))

    else:
        shape = (1, 3, 224, 224)
        transformer = caffe.io.Transformer({'data': shape})
        channel_mean = np.zeros((3, 224, 224))

    if (protocol == 'group_scene.prototxt'):
        image_mean = [90, 100, 128]  # Model specific mean

    else:
        image_mean = [104, 117, 123]  # ImageNet mean

    for channel_index, mean_val in enumerate(image_mean):
        channel_mean[channel_index, ...] = mean_val

    transformer.set_mean('data', channel_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_transpose('data', (2, 0, 1))

    return transformer


def resize_image(image, size, caffe):
    """Resize both dimensions of the image to the given size."""

    if (image.shape[0] != size) or (image.shape[1] != size):
        image = caffe.io.resize_image(image, (size, size))

    return image


def extract_features(args):
    """Caffe features extracting pipeline for PIPA and PISC datasets."""

    # Check for python correct version and set environment variables
    check_python_2()
    set_environment_caffe()
    set_environment_hdf5()

    # Get current date
    today = date.today()

    # Get feature information
    split_cue = args.cue.split('_')
    feature = split_cue[0]
    attribute = split_cue[1]

    # Get model information
    model = feature2model[args.cue]
    protocol = model2protocol[model]
    layer = model2layer[model]
    library = model2library[model]

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_metadata = os.path.join(path_dataset, args.dataset + '.json')
    path_hdf5_features = os.path.join(path_dataset, 'features', 'hdf5')
    path_images_full = os.path.join(path_dataset, 'images', 'full')
    path_splits_type = os.path.join(path_dataset, 'splits', args.type)
    path_caffe_model = os.path.join(args.path_models, 'Caffe', 'models', model)
    path_caffe_protocol = os.path.join(
        args.path_models, 'Caffe', 'protocols', protocol)
    path_caffe_library = os.path.join(args.path_caffe, library, 'python')

    # Get Caffe version and set gpu mode
    caffe = caffe_import(path_caffe_library)
    caffe_version = caffe.__version__
    caffe.set_mode_gpu()

    # Get transformer
    transformer_RGB = initialize_transformer(caffe, protocol)

    # Load model
    network = caffe.Net(path_caffe_protocol, caffe.TEST,
                        weights=path_caffe_model)

    print('>> [EXTRACTING]')
    print('>>   %s %s data...' % (args.dataset, args.type))
    print('>>     - Feature: %s %s' % (feature, attribute))
    print('>>     - Model: %s' % model)
    print('>>     - Protocol: %s' % protocol)
    print('>>     - Layer: %s' % layer)
    print('>>     - Caffe: %s' % caffe_version)

    # Load metadata
    dataset = load_json_file(path_metadata)

    for split in splits:

        # Generate work paths
        path_split = os.path.join(path_splits_type, split + '.json')
        path_save_folder = os.path.join(path_hdf5_features, args.type, split)
        path_save_file = os.path.join(path_save_folder, args.cue + '.hdf5')

        # Create folder
        if not os.path.isdir(path_save_folder):
            os.makedirs(path_save_folder)

        # Load split id list
        list_split = load_json_file(path_split)

        # Progression counter
        fraction = len(list_split) * 0.1
        porcent = 0

        print('>>     Saving %s features to: %s' % (split, path_save_file))

        with h5py.File(path_save_file, 'w') as file:

            # Save extraction metadata
            file.attrs['caffe'] = caffe_version
            file.attrs['date'] = today.strftime("%d/%m/%Y")
            file.attrs['layer'] = layer
            file.attrs['model'] = model
            file.attrs['protocol'] = protocol

            for index, id in enumerate(list_split):

                # Progression counter
                if (int(index % fraction) == 0):
                    print('>>       ...{0:.0%}'.format(porcent))
                    porcent += 0.1

                # Create a group for each image id
                group_image = file.create_group(id)

                # Load image and get size
                str_image = id.zfill(5) + '.jpg'
                path_image = os.path.join(path_images_full, str_image)
                image = caffe.io.load_image(path_image)
                height, width, channels = image.shape

                # Objects pipeline
                if (feature == 'object'):

                    for num, bounding_box in enumerate(dataset[id]['object']['bbox']):

                        # Object counter
                        str_object = str(num + 1)

                        # Fix box
                        fixed_bounding_box = fix_bounding_box(
                            bounding_box, (width, height))

                        # Get image and resize
                        image_cropped = crop_bounding_box(
                            fixed_bounding_box, image)
                        image_resized = resize_image(image_cropped, 256, caffe)

                        # Feed image and forward step
                        network.blobs['data'].data[...] = transformer_RGB.preprocess(
                            'data', image_resized)
                        probs = network.forward()

                        # Get features and logits
                        features_extracted = network.blobs[layer].data[0]
                        probs_extracted = network.blobs['prob'].data[0]

                        # Create dataset and save metadata
                        features_dataset = group_image.create_dataset(
                            str_object, data=features_extracted)
                        features_dataset.attrs['label'] = np.argmax(
                            probs_extracted)

                    # Show input image
                    if (args.show):

                        # Visualization of original input image
                        im = transformer_RGB.deprocess(
                            'data', network.blobs['data'].data[0])
                        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Input Image', imRGB)
                        cv2.waitKey(0)

                        # Visualization of processed input image
                        im2 = network.blobs['data'].data[0]
                        im2 = im2.transpose()
                        im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Processed Input Image', im2RGB)
                        cv2.waitKey(0)

                        time.sleep(8)

                    # Debug mode
                    if (args.debug):

                        print('>>       [DEBUG]')
                        print('>>         - blobs: {}\n>>         - params: {}'.format(
                            network.blobs.keys(), network.params.keys()))

                        # Shape and mean for input and extracted layers
                        print(
                            '>>         - data shape: {}'.format(network.blobs['data'].data[0].shape))
                        print('>>         - data mean: %f' %
                              network.blobs['data'].data[0].mean())
                        print('>>         - ' + layer +
                              ' shape: {}'.format(network.blobs[layer].data[0].shape))
                        print('>>         - ' + layer + ' mean: %f' %
                              network.blobs[layer].data[0].mean())

                        # If the model has a probabilities output layer
                        if ('prob' in network.blobs):
                            print(
                                '>>         - prob shape: {}'.format(network.blobs['prob'].data[0].shape))
                            print('>>         - prob mean: %f' %
                                  network.blobs['prob'].data[0].mean())
                            print('>>         - prob label %d ' %
                                  np.argmax(network.blobs['prob'].data[0]))

                        # Dataset information
                        print('>>         - dataset: {}'.format(features_dataset))
                        if ('prob' in network.blobs):
                            print(
                                '>>         - label: {}'.format(features_dataset.attrs['label']))

                # Context pipeline
                elif (feature == 'context'):

                    for relation in dataset[id][args.type]:

                        # Get relation persons boxes
                        bounding_box_person_1, bounding_box_person_2 = get_bounding_boxes(
                            relation, dataset[id]['body_bbox'])

                        # Generate context box and fix it
                        context_bounding_box = get_context_bounding_box(
                            bounding_box_person_1, bounding_box_person_2)
                        fixed_context_bounding_box = fix_bounding_box(
                            context_bounding_box, (width, height))

                        # Get image and resize
                        image_cropped = crop_bounding_box(
                            fixed_context_bounding_box, image)
                        image_resized = resize_image(image_cropped, 256, caffe)

                        # Feed image and forward step
                        network.blobs['data'].data[...] = transformer_RGB.preprocess(
                            'data', image_resized)
                        probs = network.forward()

                        # Get features
                        features_extracted = network.blobs[layer].data[0]

                        # Create dataset
                        features_dataset = group_image.create_dataset(
                            relation, data=features_extracted)

                        # Get probs and save metadata
                        if (attribute == 'emotion'):
                            probs_extracted = network.blobs['prob'].data[0]
                            features_dataset.attrs['label'] = np.argmax(
                                probs_extracted)

                    # Show input image
                    if (args.show):

                        # Visualization of original input image
                        im = transformer_RGB.deprocess(
                            'data', network.blobs['data'].data[0])
                        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Input Image', imRGB)
                        cv2.waitKey(0)

                        # Visualization of processed input image
                        im2 = network.blobs['data'].data[0]
                        im2 = im2.transpose()
                        im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                        cv2.imshow('Processed Input Image', im2RGB)
                        cv2.waitKey(0)

                        time.sleep(8)

                    # Debug mode
                    if (args.debug):

                        print('>>       [DEBUG]')
                        print('>>         - blobs: {}\n>>         - params: {}'.format(
                            network.blobs.keys(), network.params.keys()))

                        # Shape and mean for input and extracted layers
                        print(
                            '>>         - data shape: {}'.format(network.blobs['data'].data[0].shape))
                        print('>>         - data mean: %f' %
                              network.blobs['data'].data[0].mean())
                        print('>>         - ' + layer +
                              ' shape: {}'.format(network.blobs[layer].data[0].shape))
                        print('>>         - ' + layer + ' mean: %f' %
                              network.blobs[layer].data[0].mean())

                        # If the model has a probabilities output layer
                        if ('prob' in network.blobs):
                            print(
                                '>>         - prob shape: {}'.format(network.blobs['prob'].data[0].shape))
                            print('>>         - prob mean: %f' %
                                  network.blobs['prob'].data[0].mean())
                            print('>>         - prob label %d ' %
                                  np.argmax(network.blobs['prob'].data[0]))

                        # Dataset information
                        print('>>         - dataset: {}'.format(features_dataset))
                        if ('prob' in network.blobs):
                            print(
                                '>>         - label: {}'.format(features_dataset.attrs['label']))

                # Person pipeline
                elif (feature == 'body'):

                    for num, bounding_box in enumerate(dataset[id]['body_bbox']):

                        # Person counter
                        str_person = str(num + 1)

                        # Fix box
                        fixed_bounding_box = fix_bounding_box(
                            bounding_box, (width, height))

                        # Get image and resize
                        image_cropped = crop_bounding_box(
                            fixed_bounding_box, image)
                        image_resized = resize_image(image_cropped, 256, caffe)

                        # Feed image, forward step and features extraction
                        network.blobs['data'].data[...] = transformer_RGB.preprocess(
                            'data', image_resized)
                        features_extracted_1 = network.blobs[layer].data[0]

                        if (attribute != 'activity'):

                            # Feed image, forward step and features extraction
                            network.blobs['data_1'].data[...] = transformer_RGB.preprocess(
                                'data', image_resized)
                            features_extracted_2 = network.blobs[layer].data[1]

                        probs = network.forward()

                        # Create dataset
                        features_dataset = group_image.create_dataset(
                            str_person, data=features_extracted_1)

                        # Show input images
                        if (args.show):

                            # Visualization of original input images
                            im = transformer_RGB.deprocess(
                                'data', network.blobs['data'].data[0])
                            imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            cv2.imshow('Input Image 1', imRGB)
                            cv2.waitKey(0)

                            # Visualization of processed input images
                            im2 = network.blobs['data'].data[0]
                            im2 = im2.transpose()
                            im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                            cv2.imshow('Processed Input Image 1', im2RGB)
                            cv2.waitKey(0)

                            if (attribute != 'activity'):

                                # Visualization of original input images
                                im = transformer_RGB.deprocess(
                                    'data', network.blobs['data_1'].data[0])
                                imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                                cv2.imshow('Input Image 2', imRGB)
                                cv2.waitKey(0)

                                # Visualization of processed input images
                                im2 = network.blobs['data_1'].data[0]
                                im2 = im2.transpose()
                                im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                                cv2.imshow('Processed Input Image 2', im2RGB)
                                cv2.waitKey(0)

                            time.sleep(8)

                        # Debug mode
                        if (args.debug):

                            print('>>       [DEBUG]')
                            print('>>         - blobs: {}\n>>         - params: {}'.format(
                                network.blobs.keys(), network.params.keys()))

                            # Blobs and Param for important layers
                            print(
                                '>>         - data shape: {}'.format(network.blobs['data'].data[0].shape))
                            print('>>         - data mean: %f' %
                                  network.blobs['data'].data[0].mean())
                            print('>>         - data ' + layer +
                                  ' shape: {}'.format(network.blobs[layer].data[0].shape))
                            print('>>         - data ' + layer + ' mean: %f' %
                                  network.blobs[layer].data[0].mean())

                            if (attribute != 'activity'):

                                # Blobs and Param for important layers
                                print(
                                    '>>         - data_1 shape: {}'.format(network.blobs['data_1'].data[0].shape))
                                print('>>         - data_1 mean: %f' %
                                      network.blobs['data_1'].data[0].mean())
                                print('>>         - data_1 ' + layer +
                                      ' shape: {}'.format(network.blobs[layer].data[1].shape))
                                print('>>         - data_1 ' + layer + ' mean: %f' %
                                      network.blobs[layer].data[1].mean())

                            # Dataset information
                            print(
                                '>>         - dataset: {}'.format(features_dataset))

                else:
                    print('>> [ERROR] Undefined feature type')
                    sys.exit(1)

        print('>>       ...{0:.0%}'.format(porcent))

    print('>>   ... done!')
    print('>>   %s %s %s %s features extraction finished!' %
          (args.dataset, args.type, feature, attribute))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Features extracting')
    parser.add_argument('path_caffe', type=str, metavar='DIR',
                        help='path to caffe libraries')
    parser.add_argument('path_models', type=str,
                        metavar='DIR', help='path to caffe models')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])
    parser.add_argument('type',  type=str, help='type of data', choices=[
        'domain',
        'relationship'
    ])
    parser.add_argument('cue', type=str, help='type of feature', choices=[
        'body_activity',
        'body_age',
        'body_clothing',
        'body_gender',
        'context_activity',
        'context_emotion',
        'object_attention'
    ])
    parser.add_argument('--show', action='store_true',
                        help='show input images')
    parser.add_argument('--debug', action='store_true',
                        help='activate debug mode')

    args = parser.parse_args()
    extract_features(args)

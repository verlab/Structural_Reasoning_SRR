"""
    Process dataset metadata.
"""


import argparse
import os
import sys
from copy import deepcopy

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from tools.utils import (check_python_3, load_json_file, load_string_list_file,
                         save_json_file)
from tools.utils_dataset import (calculate_body_bounding_box,
                                 calculate_body_bounding_box_PIPA_relation,
                                 calculate_relations, fix_bounding_box,
                                 process_line_PIPA, process_line_PIPA_relation,
                                 relation_counter_PIPA)


def add_objects(data_processed, path_objects):
    """Add objects annotations from the given path to processed data."""

    # Progression counter
    fraction = len(data_processed) * 0.1
    porcent = 0

    for index, image in enumerate(data_processed):

        # Progression counter
        if (int(index % fraction) == 0):
            print('>>       ...{0:.0%}'.format(porcent))
            porcent += 0.1

        path_object = os.path.join(path_objects, image.zfill(5) + '.json')

        # Check if annotations for current image exists
        if (os.path.isfile(path_object)):

            # Get image size
            image_width = data_processed[image]['width']
            image_height = data_processed[image]['height']

            data_object = load_json_file(path_object)
            data_final = {}

            # Processing information
            data_final['category'] = data_object['categories']
            data_final['bbox'] = []

            # Fixing boxes
            for bounding_box in data_object['bboxes']:
                fixed_box = fix_bounding_box(
                    bounding_box, (image_width, image_height))
                data_final['bbox'].append(fixed_box)

            data_processed[image]['object'] = data_final

    print('>>       ...{0:.0%}'.format(porcent))

    return data_processed


def build_PIPA(data_indexes, path_images_full):
    """Build PIPA dataset from original annotations."""

    data_built = {}

    # Progression counter
    fraction = len(data_indexes) * 0.1
    porcent = 0

    for index, line in enumerate(data_indexes):

        # Progression counter
        if (int(index % fraction) == 0):
            print('>>       ...{0:.0%}'.format(porcent))
            porcent += 0.1

        # Getting information from each line
        str_image, original_x, original_y, original_width, original_height, identity_id, subset_id = process_line_PIPA(
            line)

        # Building original box
        original_bbox = [original_x, original_y,
                         original_width, original_height]

        # Getting image id and loading it
        path_image = os.path.join(path_images_full, str_image + '.jpg')
        image = Image.open(path_image).convert('RGB')
        image_width, image_height = image.size

        # Calculating new face coordinates
        face_left = original_x - 1
        face_upper = original_y - 1

        # Fixing negative values
        if (face_left < 0):
            face_left = 0

        if (face_upper < 0):
            face_upper = 0

        face_right = face_left + original_width + 1
        face_lower = face_upper + original_height + 1

        # Fixing face bounding boxes
        face_bbox = fix_bounding_box(
            (face_left, face_upper, face_right, face_lower), (image_width, image_height))

        # Calculating body bounding boxes as 3 * width and 6 * height
        body_bbox = calculate_body_bounding_box(original_bbox)

        # Fixing body bounding boxes
        fixed_body_bbox = fix_bounding_box(
            body_bbox, (image_width, image_height))

        # For list comparison
        fixed_body_bbox[2] -= 1
        fixed_body_bbox[3] -= 1

        # Storing information
        if str_image not in data_built:

            data_dict = {}
            data_dict['original_bbox'] = []
            data_dict['face_bbox'] = []
            data_dict['body_bbox'] = []
            data_dict['identity_id'] = []
            data_dict['subset_id'] = []
            data_dict['height'] = image_height
            data_dict['width'] = image_width
            data_dict['domain'] = {}
            data_dict['relationship'] = {}

            data_built[str_image] = data_dict

        data_built[str_image]['original_bbox'].append(original_bbox)
        data_built[str_image]['face_bbox'].append(face_bbox)
        data_built[str_image]['body_bbox'].append(fixed_body_bbox)
        data_built[str_image]['identity_id'].append(identity_id)
        data_built[str_image]['subset_id'].append(subset_id)

    print('>>       ...{0:.0%}'.format(porcent))

    return data_built


def build_PISC(
    data_image,
    data_domain,
    data_relationship,
    data_occupation
):
    """Build PISC dataset from annotations."""

    data_built = {}

    # Transforming into a dictionary
    for image in data_image:

        # Removing id from within the data
        information = image.copy()
        del information['id']

        # Using id as key for processed data
        data_built[str(image['id'])] = information

    # Processing annotations
    for image in data_built:

        # Renaming keys
        data_built[image]['height'] = data_built[image].pop('imgH')
        data_built[image]['width'] = data_built[image].pop('imgW')
        data_built[image]['original_bbox'] = data_built[image].pop('bbox')

        # List for fixed bounding boxes
        data_built[image]['body_bbox'] = []

        # Fix bounding boxes
        for box in data_built[image]['original_bbox']:
            fixed_box = fix_bounding_box(
                box, (data_built[image]['width'] - 1, data_built[image]['height'] - 1))
            data_built[image]['body_bbox'].append(fixed_box)

        # Adding social relation ajusting labels to start from zero
        if image in data_domain:
            data_built[image]['domain'] = {
                k: (v - 1) for (k, v) in data_domain[image].items()}

        if image in data_relationship:
            data_built[image]['relationship'] = {
                k: (v - 1) for (k, v) in data_relationship[image].items()}

    # Adding occupation
    for image in data_occupation:

        id = image['id']

        if id in data_built:
            data_built[id]['occupation'] = image['occupation']

    return data_built


def process_PIPA(
    data_built,
    list_data_train,
    list_data_test,
    list_data_validation,
    path_images_face,
    path_images_body,
    path_images_full
):
    """Process PIPA dataset adding relationship information from PIPA-relation."""

    dict_built = {}
    dict_splits = {'train': list_data_train,
                   'test': list_data_test, 'validation': list_data_validation}

    # Process data for each split
    # Here we also treat the wrong labeling on the PIPA-relation annotations
    for key in dict_splits:

        split = dict_splits[key]

        print('>>       ' + key.capitalize() + ' split...')

        relation_counter_actual = relation_counter_PIPA(split[0])
        relation_counter_total = {}
        list_errors = []

        # Count the total number of possible relations for the split
        for image in data_built:
            relation_counter_total[image] = calculate_relations(
                len(data_built[image]['identity_id']))

        # Compare to the total number of relations with the actual number
        # If we dont have the full number of relations, we put it on the error list
        for image in relation_counter_actual:
            if (relation_counter_total[image] != relation_counter_actual[image]):
                list_errors.append(image)

        # Progression counter
        split_size = len(split[0])
        fraction = split_size * 0.1
        porcent = 0

        for index in range(split_size):

            # Progression counter
            if (int(index % fraction) == 0):
                print('>>       ...{0:.0%}'.format(porcent))
                porcent += 0.1

            # Getting information for each person
            str_image_1, str_person_1, int_relation_1, int_domain_1 = process_line_PIPA_relation(
                split[0][index])
            str_image_2, str_person_2, int_relation_2, int_domain_2 = process_line_PIPA_relation(
                split[1][index])

            # Check if information is coherent
            assert str_image_1 == str_image_2, \
                '>> [ERROR] Image ID missmatch for relation %d' % index

            assert int_relation_1 == int_relation_2, \
                '>> [ERROR] Relation label missmatch for relation %d' % index

            assert int_domain_1 == int_domain_2, \
                '>> [ERROR] Domain label missmatch for relation %d' % index

            # Get person numbers
            person_1 = int(str_person_1[1])
            person_2 = int(str_person_2[1])

            # If the image does not have a full relation set
            # The person number from original annotation might be wrong
            # We need to find the correct bounding boxes
            if str_image_1 in list_errors:

                # Getting the full image
                path_image = os.path.join(
                    path_images_full, str_image_1 + '.jpg')
                image_full = Image.open(path_image).convert('L')

                # Getting face images from annotaded bounding boxes
                path_face_1 = os.path.join(
                    path_images_face, str_person_1 + '_' + str_image_1 + '.jpg')
                path_face_2 = os.path.join(
                    path_images_face, str_person_2 + '_' + str_image_2 + '.jpg')

                img_face_1 = np.asarray(Image.open(
                    path_face_1).resize((50, 50)).convert('L'))
                img_face_2 = np.asarray(Image.open(
                    path_face_2).resize((50, 50)).convert('L'))

                # Structural similarity
                ssim_face_1 = 0.
                ssim_face_2 = 0.

                # Here we use structural similarity to identify the correct boxes
                # For each box we crop it and check the similarity with the original image
                for i, bbox in enumerate(data_built[str_image_1]['face_bbox']):

                    cropped_face = np.asarray(image_full.crop(
                        (bbox[0], bbox[1], bbox[2], bbox[3])).resize((50, 50)))

                    ssim_cropped_1 = ssim(cropped_face, img_face_1)
                    ssim_cropped_2 = ssim(cropped_face, img_face_2)

                    if (ssim_face_1 < ssim_cropped_1):
                        ssim_face_1 = ssim_cropped_1
                        person_1 = i + 1

                    if (ssim_face_2 < ssim_cropped_2):
                        ssim_face_2 = ssim_cropped_2
                        person_2 = i + 1

            # Getting the full image
            path_image = os.path.join(path_images_full, str_image_1 + '.jpg')
            image_full = Image.open(path_image).convert('L')

            # Getting face and body images for checking
            path_face_1 = os.path.join(
                path_images_face, str_person_1 + '_' + str_image_1 + '.jpg')
            path_face_2 = os.path.join(
                path_images_face, str_person_2 + '_' + str_image_2 + '.jpg')

            path_body_1 = os.path.join(
                path_images_body, str_person_1 + '_' + str_image_1 + '.jpg')
            path_body_2 = os.path.join(
                path_images_body, str_person_2 + '_' + str_image_2 + '.jpg')

            img_face_1 = Image.open(path_face_1).convert('L')
            img_face_2 = Image.open(path_face_2).convert('L')

            img_body_1 = Image.open(path_body_1).convert('L')
            img_body_2 = Image.open(path_body_2).convert('L')

            # Crop face and body from boxes to compare
            body_box_1 = calculate_body_bounding_box_PIPA_relation(
                data_built[str_image_1]['original_bbox'][person_1 - 1])
            body_box_2 = calculate_body_bounding_box_PIPA_relation(
                data_built[str_image_1]['original_bbox'][person_2 - 1])

            fixed_body_box_1 = fix_bounding_box(body_box_1, image_full.size)
            fixed_body_box_2 = fix_bounding_box(body_box_2, image_full.size)

            cropped_face_1 = image_full.crop(
                data_built[str_image_1]['face_bbox'][person_1 - 1])
            cropped_face_2 = image_full.crop(
                data_built[str_image_2]['face_bbox'][person_2 - 1])

            cropped_body_1 = image_full.crop(fixed_body_box_1)
            cropped_body_2 = image_full.crop(fixed_body_box_2)

            # Build the string key for the relation
            str_relation = str(person_1) + ' ' + str(person_2)

            # Check if this is not an spurious relation
            assert str_relation not in data_built[str_image_1]['relationship'], \
                '>> [ERROR] Spurious relation for image %s' % str_image_1

            # Check if the boxes for person 1 are correct
            assert (img_face_1.size == cropped_face_1.size) and (img_body_1.size == cropped_body_1.size), \
                '>> [ERROR] Bounding Box 1 missmatch for image %s' % str_image_1

            # Check if the boxes for person 2 are correct
            assert (img_face_2.size == cropped_face_2.size) and (img_body_2.size == cropped_body_2.size), \
                '>> [ERROR] Bounding Box 2 missmatch for image %s' % str_image_2

            # Add relation and domain information
            data_built[str_image_1]['relationship'][str_relation] = int_relation_1
            data_built[str_image_1]['domain'][str_relation] = int_domain_1

        print('>>       ...{0:.0%}'.format(porcent))

        # Generating split lists
        dict_built[key] = list(relation_counter_actual.keys())

    return data_built, dict_built


def pre_process(args):
    """Processing pipeline for PIPA and PISC metadata."""

    # Check for python correct version
    check_python_3()

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_save_built = os.path.join(
        path_dataset, 'built_' + args.dataset + '.json')
    path_save_final = os.path.join(path_dataset, args.dataset + '.json')
    path_annotations = os.path.join(path_dataset, 'annotations')
    path_objects = os.path.join(path_dataset, 'objects')
    path_splits = os.path.join(path_dataset, 'splits')
    path_images = os.path.join(path_dataset, 'images')
    path_images_full = os.path.join(path_images, 'full')
    path_images_face = os.path.join(path_images, 'face')
    path_images_body = os.path.join(path_images, 'body')
    path_split_domain = os.path.join(path_splits, 'domain')
    path_split_relationship = os.path.join(path_splits, 'relationship')

    if args.dataset == 'PIPA':

        print('>> [PRE-PROCESSING]')
        print('>>   People in Photo Album (PIPA) relation dataset')

        # Generating PIPA specific paths
        path_train_1 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body1_train_16.txt')
        path_train_2 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body2_train_16.txt')
        # Test and validation annotations original file names have been switched!
        path_test_1 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body1_eval_16.txt')
        # Test and validation annotations original file names have been switched!
        path_test_2 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body2_eval_16.txt')
        # Test and validation annotations original file names have been switched!
        path_validation_1 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body1_test_16.txt')
        # Test and validation annotations original file names have been switched!
        path_validation_2 = os.path.join(
            path_splits, 'consistency_annotator', 'single_body2_test_16.txt')
        path_data_indexes = os.path.join(
            path_annotations, 'original', 'index.txt')

        print('>>     Loading dataset from: %s' % path_dataset)

        # Loading data
        data_indexes = load_string_list_file(path_data_indexes)
        data_train_1 = load_string_list_file(path_train_1)
        data_train_2 = load_string_list_file(path_train_2)
        data_test_1 = load_string_list_file(path_test_1)
        data_test_2 = load_string_list_file(path_test_2)
        data_validation_1 = load_string_list_file(path_validation_1)
        data_validation_2 = load_string_list_file(path_validation_2)

        # Generating data lists
        list_data_train = [data_train_1, data_train_2]
        list_data_test = [data_test_1, data_test_2]
        list_data_validation = [data_validation_1, data_validation_2]

        print('>>     Building metada...')

        # Building PIPA metadata
        data_built = build_PIPA(data_indexes, path_images_full)

        print('>>     ...done!')
        print('>>     Adding objects information...')

        # Adding objects PIPA metadata
        data_built = add_objects(data_built, path_objects)

        print('>>     ...done!')
        print('>>     Saving built data to: %s' % path_save_built)

        # Saving built data
        save_json_file(path_save_built, data_built, args.ordered)

        print('>>     Processing metada...')

        # Processing PIPA metadata
        data_built, dict_splits = process_PIPA(
            data_built, list_data_train, list_data_test, list_data_validation, path_images_face, path_images_body, path_images_full)

        print('>>     ...done!')
        print('>>     Saving final data to: %s' % path_save_final)

        # Saving metadata
        save_json_file(path_save_final, data_built, args.ordered)

        # Saving split lists
        for split in dict_splits:

            path_save_split_domain = os.path.join(
                path_split_domain, split + '.json')
            path_save_split_relationship = os.path.join(
                path_split_relationship, split + '.json')
            print('>>     Saving ' + split + ' data to: %s' %
                  path_save_split_domain)
            print('>>     Saving ' + split + ' data to: %s' %
                  path_save_split_relationship)

            save_json_file(path_save_split_domain,
                           dict_splits[split], args.ordered)
            save_json_file(path_save_split_relationship,
                           dict_splits[split], args.ordered)

        print('>>   PIPA dataset processing finished!')

    else:

        print('>> [PRE-PROCESSING]')
        print('>>   People in Social Context (PISC) dataset')

        # Generating PISC specific paths
        path_data_image = os.path.join(path_annotations, 'image_info.json')
        path_data_domain = os.path.join(path_annotations, 'domain.json')
        path_data_relationship = os.path.join(
            path_annotations, 'relationship.json')
        path_data_occupation = os.path.join(
            path_annotations, 'occupation.json')

        print('>>     Loading dataset from: %s' % path_dataset)

        # Loading data
        data_image = load_json_file(path_data_image)
        data_domain = load_json_file(path_data_domain)
        data_relationship = load_json_file(path_data_relationship)
        data_occupation = load_json_file(path_data_occupation)

        print('>>     Building metadata...')

        # Building PISC metadata
        data_built = build_PISC(data_image, data_domain,
                                data_relationship, data_occupation)

        print('>>     ...done!')
        print('>>     Adding objects information...')

        # Adding objects PISC metadata
        data_final = add_objects(data_built, path_objects)

        print('>>     ...done!')
        print('>>     Saving final data to: %s' % path_save_final)

        # Saving final data
        save_json_file(path_save_final, data_final, args.ordered)

        print('>>   PISC dataset processing finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset pre-processing')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])
    parser.add_argument('--ordered', action='store_true',
                        help='save json ordered')

    args = parser.parse_args()
    pre_process(args)

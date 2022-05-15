"""
    Tests for metadata.
"""


import argparse
import os

from PIL import Image

from tools.utils import (check_python_2, check_python_3, load_json_file,
                         load_string_list_file, save_string_list_file)
from tools.utils_dataset import (build_list_split, fix_bounding_box,
                                 process_line_list, process_line_PIPA)


def compare_lists(list_ground_truth, list_generated):
    """Compare generated list against the ground_truth."""

    # Check data size
    assert len(list_ground_truth) == len(list_generated), \
        '>> [ERROR] Lists size missmatch\n>> [ERROR]   Source: %d\n>> [ERROR]   Processed: %d' % (
            len(list_ground_truth), len(list_generated))

    # Compare line by line
    for index in range(len(list_generated)):

        # Process lines from both lists
        ground_truth_str_image, ground_truth_bbox_person_1, ground_truth_bbox_person_2, ground_truth_int_relation, ground_truth_int_domain = process_line_list(
            list_ground_truth[index])
        generated_str_image, generated_bbox_person_1, generated_bbox_person_2, generated_int_relation, generated_int_domain = process_line_list(
            list_generated[index])

        # Check if information is coherent
        assert ground_truth_str_image == generated_str_image, \
            '>> [ERROR] Image ID missmatch for relation %d' % (index + 1)

        assert ground_truth_bbox_person_1 == generated_bbox_person_1, \
            '>> [ERROR] Bounding Box 1 missmatch for relation %d' % (index + 1)

        assert ground_truth_bbox_person_2 == generated_bbox_person_2, \
            '>> [ERROR] Bounding Box 2 missmatch for relation %d' % (index + 1)

        assert ground_truth_int_relation == generated_int_relation, \
            '>> [ERROR] Relation missmatch for relation %d' % (index + 1)

        assert ground_truth_int_domain == generated_int_domain, \
            '>> [ERROR] Domain missmatch for relation %d' % (index + 1)


def test_PIPA(
    data_processed,
    data_indexes,
    path_images_full,
    path_objects
):
    """Test PIPA generated data against source annotations and other tests."""

    index_counter = {}

    # Progression counter
    fraction = len(data_indexes) * 0.1
    porcent = 0

    # Compare the processed data individually
    for num, line in enumerate(data_indexes):

        # Progression counter
        if (int(num % fraction) == 0):
            print('>>     ...{0:.0%}'.format(porcent))
            porcent += 0.1

        # Getting information from each line
        str_image, original_x, original_y, original_width, original_height, identity_id, subset_id = process_line_PIPA(
            line)

        # Update index for the order the actual box appear in the images list
        if (str_image in index_counter):
            index_counter[str_image] += 1

        else:
            index_counter[str_image] = 0

        # Getting image id and loading it
        path_image = os.path.join(path_images_full, str_image + '.jpg')
        image = Image.open(path_image).convert('L')
        image_width, image_height = image.size

        original_bbox = [original_x, original_y,
                         original_width, original_height]

        # Source comparison
        assert identity_id in data_processed[str_image]['identity_id'], \
            '>> [ERROR] Identity ID missmatch for image %s' % str_image

        assert subset_id in data_processed[str_image]['subset_id'], \
            '>> [ERROR] Subset ID missmatch for image %s' % str_image

        assert image_height == data_processed[str_image]['height'], \
            '>> [ERROR] Height missmatch for image %s' % str_image

        assert image_width == data_processed[str_image]['width'], \
            '>> [ERROR] Width missmatch for image %s' % str_image

        assert original_bbox == data_processed[str_image]['original_bbox'][index_counter[str_image]], \
            '>> [ERROR] Original bounding box missmatch for image %s' % str_image

        # Body tests
        for bounding_box in data_processed[str_image]['body_bbox']:
            fixed_box = fix_bounding_box(
                bounding_box, (image_width, image_height))

            assert bounding_box == fixed_box, \
                '>> [ERROR] Bad body bounding box coordinate for image %s' % str_image

        path_objects_image = os.path.join(
            path_objects, str_image.zfill(5) + '.json')

        # Object tests
        if (os.path.isfile(path_objects_image)):

            assert 'object' in data_processed[str_image], \
                '>> [ERROR] Objects boxes not processed for image %s' % str_image

            assert len(data_processed[str_image]['object']['category']) == len(data_processed[str_image]['object']['bbox']), \
                '>> [ERROR] Objects and categories sizes missmatch for image %s' % str_image

            # Loading objects
            data_objects = load_json_file(path_objects_image)

            for i, bounding_box in enumerate(data_processed[str_image]['object']['bbox']):

                fixed_box = fix_bounding_box(
                    bounding_box, (image_width, image_height))

                assert bounding_box == fixed_box, \
                    '>> [ERROR] Bad object bounding box coordinate for image %s' % str_image

                fixed_original_box = fix_bounding_box(
                    data_objects['bboxes'][i], (image_width, image_height))

                assert bounding_box == fixed_original_box, \
                    '>> [ERROR] Object bounding box missmatch for image %s' % str_image

    print('>>     ...{0:.0%}'.format(porcent))


def test_PISC(
    data_processed,
    data_image,
    data_domain,
    data_relationship,
    data_occupation,
    path_objects
):
    """Test PISC generated data against source annotations and other tests."""

    # Check data size
    assert len(data_processed) == len(data_image), \
        '>> [ERROR] Data size does not match\n>> [ERROR]   Source: %d\n>> [ERROR]   Processed: %d' % (
            len(data_image), len(data_processed))

    # Compare the processed data individually
    for image in data_image:

        str_image = str(image['id'])

        # Source comparison
        assert image['source'] == data_processed[str_image]['source'], \
            '>> [ERROR] Source missmatch for image %s' % str_image

        assert image['source_id'] == data_processed[str_image]['source_id'], \
            '>> [ERROR] Source ID missmatch for image %s' % str_image

        assert image['imgH'] == data_processed[str_image]['height'], \
            '>> [ERROR] Height missmatch for image %s' % str_image

        assert image['imgW'] == data_processed[str_image]['width'], \
            '>> [ERROR] Width missmatch for image %s' % str_image

        assert image['bbox'] == data_processed[str_image]['original_bbox'], \
            '>> [ERROR] Body bounding box missmatch for image %s' % str_image

        # Body tests
        for bounding_box in data_processed[str_image]['body_bbox']:
            fixed_box = fix_bounding_box(
                bounding_box, (image['imgW'], image['imgH']))

            assert bounding_box == fixed_box, \
                '>> [ERROR] Bad body bounding box coordinate for image %s' % str_image

        path_objects_image = os.path.join(
            path_objects, str_image.zfill(5) + '.json')

        # Object tests
        if (os.path.isfile(path_objects_image)):

            assert 'object' in data_processed[str_image], \
                '>> [ERROR] Objects boxes not processed for image %s' % str_image

            assert len(data_processed[str_image]['object']['category']) == len(data_processed[str_image]['object']['bbox']), \
                '>> [ERROR] Objects and categories sizes missmatch for image %s' % str_image

            # Loading objects
            data_objects = load_json_file(path_objects_image)

            for i, bounding_box in enumerate(data_processed[str_image]['object']['bbox']):

                fixed_box = fix_bounding_box(
                    bounding_box, (image['imgW'], image['imgH']))

                assert bounding_box == fixed_box, \
                    '>> [ERROR] Bad object bounding box coordinate for image %s' % str_image

                fixed_original_box = fix_bounding_box(
                    data_objects['bboxes'][i], (image['imgW'], image['imgH']))

                assert bounding_box == fixed_original_box, \
                    '>> [ERROR] Object bounding box missmatch for image %s' % str_image

    # Fix labels and compare with source
    for str_image in data_processed:

        if 'domain' in data_processed[str_image]:
            assert data_domain[str_image] == {k: v + 1 for (k, v) in data_processed[str_image]['domain'].items()}, \
                '>> [ERROR] Domain missmatch for image %s' % str_image

        if 'relationship' in data_processed[str_image]:
            assert data_relationship[str_image] == {k: v + 1 for (k, v) in data_processed[str_image]['relationship'].items()}, \
                '>> [ERROR] Relationship missmatch for image %s' % str_image

    # Compare occupation with source
    for image in data_occupation:

        str_image = image['id']

        if str_image in data_processed:
            assert image['occupation'] == data_processed[str_image]['occupation'], \
                '>> [ERROR] Occupation missmatch for image %s' % str_image


def test_metadata(args):
    """Testing pipeline for PIPA and PISC processed metadata."""

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_annotations = os.path.join(path_dataset, 'annotations')
    path_objects = os.path.join(path_dataset, 'objects')
    path_images = os.path.join(path_dataset, 'images')
    path_images_full = os.path.join(path_images, 'full')
    path_splits = os.path.join(path_dataset, 'splits')
    path_split_domain = os.path.join(path_splits, 'domain')
    path_split_relationship = os.path.join(path_splits, 'relationship')
    path_lists = os.path.join(path_splits, 'lists')
    path_processed_data = os.path.join(path_dataset, args.dataset + '.json')

    if args.dataset == 'PIPA':

        # Check for python correct version
        check_python_3()

        print('>> [TESTING]')
        print('>>   People in Photo Album (PIPA) relation dataset')

        # Generating PIPA specific paths
        path_data_indexes = os.path.join(
            path_annotations, 'original', 'index.txt')

        path_split_relationship_train = os.path.join(
            path_split_relationship, 'train.json')
        path_split_relationship_test = os.path.join(
            path_split_relationship, 'test.json')
        path_split_relationship_validation = os.path.join(
            path_split_relationship, 'validation.json')

        path_list_relationship_train = os.path.join(
            path_lists, 'relationship_train.txt')
        path_list_relationship_test = os.path.join(
            path_lists, 'relationship_test.txt')
        path_list_relationship_validation = os.path.join(
            path_lists, 'relationship_validation.txt')

        path_save_list_relationship_train = os.path.join(
            path_lists, 'generated_relationship_train.txt')
        path_save_list_relationship_test = os.path.join(
            path_lists, 'generated_relationship_test.txt')
        path_save_list_relationship_validation = os.path.join(
            path_lists, 'generated_relationship_validation.txt')

        print('>>     Loading dataset from: %s' % path_dataset)
        print('>>     Loading processed data from: %s' % path_processed_data)

        # Loading data
        data_indexes = load_string_list_file(path_data_indexes)
        data_processed = load_json_file(path_processed_data)

        split_relationship_train = load_json_file(
            path_split_relationship_train)
        split_relationship_test = load_json_file(path_split_relationship_test)
        split_relationship_validation = load_json_file(
            path_split_relationship_validation)

        list_relationship_train = load_string_list_file(
            path_list_relationship_train)
        list_relationship_test = load_string_list_file(
            path_list_relationship_test)
        list_relationship_validation = load_string_list_file(
            path_list_relationship_validation)

        print('>>     Testing against source data...')

        # Testing PIPA processed data
        test_PIPA(data_processed, data_indexes, path_images_full, path_objects)

        print('>>     ...done!')
        print('>>     Testing against ground truth...')

        # Generating lists
        generated_relationship_train = build_list_split(
            data_processed, split_relationship_train, 'relationship')
        generated_relationship_test = build_list_split(
            data_processed, split_relationship_test, 'relationship')
        generated_relationship_validation = build_list_split(
            data_processed, split_relationship_validation, 'relationship')

        # Comparing against ground truth
        compare_lists(list_relationship_train, generated_relationship_train)
        print('>>       Relationship train data OK!')

        compare_lists(list_relationship_test, generated_relationship_test)
        print('>>       Relationship test data OK!')

        compare_lists(list_relationship_validation,
                      generated_relationship_validation)
        print('>>       Relationship validation data OK!')
        print('>>     ...done!')
        print('>>     Saving generated lists to: %s' % path_lists)

        # Saving generated lists
        save_string_list_file(
            path_save_list_relationship_train, generated_relationship_train)
        save_string_list_file(
            path_save_list_relationship_test, generated_relationship_test)
        save_string_list_file(
            path_save_list_relationship_validation, generated_relationship_validation)

        print('>>   PIPA dataset testing finished!')

    else:

        # Check for python correct version
        check_python_2()

        print('>> [TESTING]')
        print('>>   People in Social Context (PISC) dataset')

        # Generating PISC specific paths
        path_data_image = os.path.join(path_annotations, 'image_info.json')
        path_data_domain = os.path.join(path_annotations, 'domain.json')
        path_data_relationship = os.path.join(
            path_annotations, 'relationship.json')
        path_data_occupation = os.path.join(
            path_annotations, 'occupation.json')

        path_split_domain_test = os.path.join(path_split_domain, 'test.json')
        path_split_relationship_test = os.path.join(
            path_split_relationship, 'test.json')

        path_list_domain_test = os.path.join(path_lists, 'domain_test.txt')
        path_list_relationship_test = os.path.join(
            path_lists, 'relationship_test.txt')

        path_save_list_domain_test = os.path.join(
            path_lists, 'generated_domain_test.txt')
        path_save_list_relationship_test = os.path.join(
            path_lists, 'generated_relationship_test.txt')

        print('>>     Loading dataset from: %s' % path_dataset)
        print('>>     Loading processed data from: %s' % path_processed_data)

        # Loading data
        data_processed = load_json_file(path_processed_data)
        data_image = load_json_file(path_data_image)
        data_domain = load_json_file(path_data_domain)
        data_relationship = load_json_file(path_data_relationship)
        data_occupation = load_json_file(path_data_occupation)

        split_domain_test = load_json_file(path_split_domain_test)
        split_relationship_test = load_json_file(path_split_relationship_test)

        list_domain_test = load_string_list_file(path_list_domain_test)
        list_relationship_test = load_string_list_file(
            path_list_relationship_test)

        print('>>     Testing against source data...')

        # Testing PISC processed data
        test_PISC(data_processed, data_image, data_domain,
                  data_relationship, data_occupation, path_objects)

        print('>>     ...done!')
        print('>>     Testing against ground truth...')

        # Generating lists
        generated_domain_test = build_list_split(
            data_processed, split_domain_test, 'domain')
        generated_relationship_test = build_list_split(
            data_processed, split_relationship_test, 'relationship')

        # Comparing against ground truth
        compare_lists(list_domain_test, generated_domain_test)
        print('>>       Domain test data OK!')

        # Comparing against ground truth
        compare_lists(list_relationship_test, generated_relationship_test)
        print('>>       Relationship test data OK!')

        print('>>     ...done!')
        print('>>     Saving generated lists to: %s' % path_lists)

        # Saving generated lists
        save_string_list_file(path_save_list_domain_test,
                              generated_domain_test)
        save_string_list_file(
            path_save_list_relationship_test, generated_relationship_test)

        print('>>   PISC dataset testing finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Metadata testing')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])

    args = parser.parse_args()
    test_metadata(args)

"""
    Process dataset images.
"""


import argparse
import os
from shutil import copy2

from tools.utils import check_python_3, load_json_file
from tools.utils_dataset import relations as relation_types
from tools.utils_dataset import splits


def process_partitions(args):
    """Partition the dataset images by number of relations, add distinct relation classes to the end of the image name."""

    # Check for python correct version and set environment variables
    check_python_3()

    # Generating paths
    path_dataset = os.path.join(args.path_datasets, args.dataset)
    path_splits = os.path.join(path_dataset, 'splits')
    path_images = os.path.join(path_dataset, 'images')
    path_images_full = os.path.join(path_images, 'full')
    path_save = os.path.join(path_images, 'processed')
    path_metadata = os.path.join(path_dataset, args.dataset + '.json')

    print('>> [PROCESSING PARTITIONS]')

    # Loading data
    data_metadata = load_json_file(path_metadata)

    # Build images partitions
    for type in relation_types:

        print('>>   Processing %s %s images...' % (args.dataset, type))

        # Create relation folder
        path_save_type = os.path.join(path_save, type)

        if not os.path.isdir(path_save_type):
            os.makedirs(path_save_type)

        # Get path
        path_split_type = os.path.join(path_splits, type)

        for split in splits:

            # Create split folder
            path_save_split = os.path.join(path_save_type, split)

            if not os.path.isdir(path_save_split):
                os.makedirs(path_save_split)

            # Loading split data
            path_split = os.path.join(path_split_type, split + '.json')
            data_split = load_json_file(path_split)

            print('>>     Creating %s partition...' % split)

            # Progression counter
            fraction = len(data_split) * 0.1
            porcent = 0

            for index, id in enumerate(data_split):

                # Progression counter
                if (int(index % fraction) == 0):
                    print('>>       ...{0:.0%}'.format(porcent))
                    porcent += 0.1

                # Get relation metadata
                relations = data_metadata[id][type]

                # Reset counter
                counter_relation = 0

                # New name
                str_name_image = id.zfill(5)

                # Get relations
                for relation in relations:

                    # Get relation number
                    number_relation = relations[relation]

                    # Add relation to string
                    str_name_image = str_name_image + \
                        '_' + str(number_relation)

                    # Increment relation counter
                    counter_relation += 1

                # Create partition folder
                path_save_partition = os.path.join(
                    path_save_split, str(counter_relation))

                if not os.path.isdir(path_save_partition):
                    os.makedirs(path_save_partition)

                # Get paths original image and new image
                path_original_image = os.path.join(
                    path_images_full, id.zfill(5) + '.jpg')
                path_save_image = os.path.join(
                    path_save_partition, str_name_image + '.jpg')

                # Copy image to the new destination
                copy2(path_original_image, path_save_image)

            print('>>     ... done!')

        print('>>   ... done!')

    print('>> Dataset partitioning finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Features testing')
    parser.add_argument('path_datasets', type=str, help='path to datasets')
    parser.add_argument('dataset', type=str, help='datasets to process', choices=[
        'PIPA',
        'PISC'
    ])

    args = parser.parse_args()
    process_partitions(args)

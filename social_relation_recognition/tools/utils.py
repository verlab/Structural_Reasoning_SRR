"""
    General routines and configurations.
"""


import json
import os
import sys

import h5py
import psutil


def check_python_2():
    """Check if it is running on Python 2."""

    assert sys.version_info[0] == 2, \
        '>> [ERROR] Must be running on Python 3'

    return


def check_python_3():
    """Check if it is running on Python 3."""

    assert sys.version_info[0] == 3, \
        '>> [ERROR] Must be running on Python 3'

    return


def get_memory():
    """Returns a string containing the use of virtual memory."""

    used_memory = psutil.virtual_memory().used
    str_memory = '{} MB'.format(used_memory/1024/1024)

    return str_memory


def get_time(time):
    """Receive time in seconds and returns an hour formatted string."""

    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    str_time = '{:0>2}:{:0>2}:{:0>2}'.format(
        int(hours), int(minutes), int(seconds))

    return str_time


def load_hdf5_dataset(path):
    """Load the hdf5 file in dataset format from the given path."""

    with h5py.File(path, 'r') as file:

        file_data = {}
        file_data['data'] = {}
        file_data['attrs'] = {}

        if (file.attrs.keys()):
            for attribute in file.attrs.keys():
                file_data['attrs'][attribute] = file.attrs[attribute]

        for id_image in file.keys():

            data_image = file[id_image]

            file_data['data'][id_image] = {}
            file_data['data'][id_image]['data'] = {}
            file_data['data'][id_image]['attrs'] = {}

            if (data_image.attrs.keys()):
                for attribute in file.attrs.keys():
                    file_data['data'][id_image]['attrs'][attribute] = file.attrs[attribute]

            for id_data in data_image.keys():

                file_data['data'][id_image]['data'][id_data] = {}
                file_data['data'][id_image]['data'][id_data]['attrs'] = {}

                file_data['data'][id_image]['data'][id_data]['data'] = data_image[id_data][()]
                file_data['data'][id_image]['data'][id_data]['size'] = data_image[id_data].size
                file_data['data'][id_image]['data'][id_data]['dtype'] = data_image[id_data].dtype

                if (data_image[id_data].attrs.keys()):
                    for attribute in data_image[id_data].attrs.keys():

                        file_data['data'][id_image]['data'][id_data]['attrs'][attribute] = data_image[id_data].attrs[attribute]

    return file_data


def load_hdf5_graph(path, prefix):
    """Load the hdf5 file in graph format from the given path."""

    with h5py.File(path, 'r') as file:

        graph_data = {}

        for id_image in file.keys():

            data_image = file[id_image]

            for id_data in data_image.keys():

                id_node = prefix + '_' + id_image + '_' + id_data
                graph_data[id_node] = data_image[id_data][()]

    return graph_data


def load_json_file(path):
    """Load the json object from the given path."""

    with open(path, 'r') as file:
        data = json.load(file)

    return data


def load_string_list_file(path):
    """Load the string list object from the given path."""

    with open(path, 'r') as file:
        list = file.readlines()

    return list


def print_hdf5_dataset(dataset, number_print):

    if (dataset['attrs']):

        print('>>  Dataset Attributes:')
        for attribute in dataset['attrs']:
            print('>>    - ' + attribute + ':', dataset['attrs'][attribute])

    for number, image in enumerate(dataset['data']):
        image_data = dataset['data'][image]

        if (image_data['attrs']):

            print('>>    Image Attributes:')
            for attribute in image_data['attrs']:
                print('>>      - ' + attribute + ':',
                      image_data['attrs'][attribute])

        for key in image_data['data'].keys():
            print('>>    Image: %s\n>>      Key: %s' % (image, key))
            print('>>        - data:', image_data['data'][key]['data'][:5])
            print('>>        - size:', image_data['data'][key]['size'])
            print('>>        - dtype:', image_data['data'][key]['dtype'])

            if (image_data['data'][key]['attrs']):

                print('>>      Attributes:')
                for attribute in image_data['data'][key]['attrs']:
                    print('>>      - ' + attribute + ':',
                          image_data['data'][key]['attrs'][attribute])

        if ((number + 1) == number_print):
            return


def save_json_file(path, data, sort):
    """Save the json object to the given path."""

    with open(path, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=sort)


def save_string_list_file(path, list):
    """Save the string list object to the given path."""

    size = len(list)

    with open(path, 'w') as file:
        for index, item in enumerate(list):
            file.write(item)

            if (index < size - 1):
                file.write('\n')


def set_environment_caffe():
    """Set environment variable for Caffe verbosity."""

    # To reduce Caffe verbosity
    os.environ['GLOG_minloglevel'] = '1'


def set_environment_hdf5():
    """Set environment variable for hdf5 file locking."""

    # To overwrite hdf5 files
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def set_environment_tensorflow():
    """Set environment variable for Tensorflow verbosity."""

    # To reduce Tensorflow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def set_environment_dgl_backend(backend):
    """Set environment variable for DGL backend."""

    # To set DGL backend framework
    os.environ['DGLBACKEND'] = backend


def set_environment_cuda():
    """Set environment variable for CUDA reproducibility."""

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

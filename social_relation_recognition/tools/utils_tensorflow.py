"""
    Tensorflow related routines and configurations.
"""


import os
import sys

import tensorflow


def sanity_check():
    """Check Python version, environment variables, Tensorflow version and GPU devices."""

    print('>> [SANITY CHECK]')
    print('>>   Environment Variables: ')
    print('>>     - HDF5_USE_FILE_LOCKING:',
          os.environ['HDF5_USE_FILE_LOCKING'])
    print('>>     - TF_CPP_MIN_LOG_LEVEL:', os.environ['TF_CPP_MIN_LOG_LEVEL'])
    print('>>   Python Version: %d.%d.%d' %
          (sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('>>   Tensorflow Version:', tensorflow.version.VERSION)

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')

    if (gpus):
        try:
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tensorflow.config.experimental.list_logical_devices(
                'GPU')
            print('>>   Devices: ')
            print('>>     -', len(gpus), 'Physical GPU(s)')
            print('>>     -', len(logical_gpus), 'Logical GPU(s)')
        except RuntimeError as e:
            print(e)

    else:
        print('>> [ERROR] No GPU(s) device(s) available')
        sys.exit()

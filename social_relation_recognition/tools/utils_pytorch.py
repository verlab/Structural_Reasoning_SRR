"""
    Pytorch related routines and configurations.
"""


import os
import sys

import torch


def sanity_check():
    """Check environment variables, available GPU devices, software version and modes."""

    print('>> [SANITY CHECK]')
    print('>>   Environment Variables: ')
    print('>>     - DGLBACKEND:', os.environ['DGLBACKEND'])
    print('>>   Devices:')
    print('>>     -', torch.cuda.device_count(), 'Physical GPU(s) available')
    print('>>   Versions: ')
    print('>>     - Python: %d.%d.%d' %
          (sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('>>     - Pytorch:', torch.__version__)
    print('>>     - Cuda:', torch.version.cuda)
    print('>>     - cuDNN:', torch.backends.cudnn.version())
    print('>>   Modes:')
    print('>>     - Deterministic:', torch.backends.cudnn.deterministic)
    print('>>     - Benchmark:', torch.backends.cudnn.benchmark)

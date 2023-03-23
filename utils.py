"""
Some common functions for this benchmark suite.

Mailto: xmei@jlab.org
"""

import socket
import torch


class FCParamConfigs:
    """
    Hyperparameters are taken from Table 2 in https://yuemmawang.github.io/publications/wang-mlsys2020.pdf.
    """

    class Layers:
        MIN = 4
        MAX = 128

    class Nodes:
        MIN = 32
        MAX = 8192

    class BatchSize:
        MIN = 64
        MAX = 16384

    class InputSize:
        MIN = 2000
        EX_MAX = 10000
        INC = 2000

    class OutputSize:
        MIN = 200
        EX_MAX = 1200
        INC = 200

    class AdamParams:
        EPS = 1e-8  # 1e-7 for TensorFlow


def get_device_name(flag):
    """
    Get the device name.
    @param flag is for using
    """
    if torch.cuda.is_available() and flag:
        return torch.cuda.get_device_name()
    else:
        return socket.gethostname() + "-cpu"

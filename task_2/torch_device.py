# encoding: utf-8

"""
Get a PyTorch device to be used for running
code on, prefers gpu over cpu.
"""

import torch


def get_device() -> torch.device:
    """
    Returns gpu device for PyTorch if
    available, otherwise cpu.

    :return: torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

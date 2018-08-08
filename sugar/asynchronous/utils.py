"""
Utility methods.
"""

import torch
import torch.distributed as dist


def is_master():
    return dist.get_rank() == 0

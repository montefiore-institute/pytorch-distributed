import torch
import torch.distributed as dist


def is_master():
    r"""Utility function that checks if the process holds the master rank.

    This method will evaluate to true when ``torch.distributed.get_rank() == 0``.
    """
    return dist.get_rank() == 0

"""
Gradient Energy Matching
"""

import torch
import torch.distributed as dist

from torch.distributed import get_rank, get_world_size

from .optimizer import AsynchronousOptimizer, required
from .utils import is_master



class GEMGlobal(AsynchronousOptimizer):
    """
    GEM with global proxy estimation.
    """

    def __init__(self, params, lr=0.01, momentum=.9, epsilon=10e-8, workers=None):
        defaults = dict(lr=lr, momentum=momentum, epsilon=epsilon)
        super(GEMGlobal, self).__init__(params, defaults, workers)
        self._num_workers = len(self.workers())
        self._proxy_fraction = 1. / self._num_workers

    def _allocate_master_resources(self):
        # Allocate the central variable buffers.
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["buffer"] = torch.zeros_like(p)
                param_state["proxy"] = torch.zeros_like(p)
                param_state["stale"] = {}
                for worker in self.workers():
                    param_state["stale"][worker] = p.clone()

    def _master_procedure(self):
        # Receive the rank of the next worker in the queue.
        worker_rank = self._next_worker()
        # Receive the update and store it in the parameter buffer.
        requests = []
        for group in self.param_groups:
            for p in group["params"]:
                parameter_buffer = self.state[p]["buffer"]
                request = dist.irecv(parameter_buffer.data, src=worker_rank)
                requests.append(request)
        # Apply the GEM procedure, and send the parameters.
        for group in self.param_groups:
            epsilon = group["epsilon"]
            momentum = group["momentum"]
            for index, p in enumerate(group["params"]):
                state = self.state[p]
                parameter_buffer = state["buffer"]
                proxy = state["proxy"]
                stale = state["stale"][worker_rank]
                requests[index].wait()
                # Check if the proxy needs to be updated.
                # Original method:
                # if torch.rand(1, requires_grad=False).item() < self._proxy_fraction:
                #     proxy.data *= momentum
                #     proxy.data += parameter_buffer.data
                # Alternative:
                proxy.data *= momentum
                proxy.data += parameter_buffer.data / (self._num_workers / 2)
                pi = (proxy.data.abs() - (p.data - stale.data)) / (parameter_buffer.data.abs() + epsilon)
                pi.clamp_(min=0., max=10.)
                parameter_buffer *= pi
                p.data.add_(parameter_buffer.data)
                dist.isend(p.data, dst=worker_rank)
        # Set the worker staleness factor.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["stale"][worker_rank].data.copy_(p.data)

    def _worker_procedure(self):
        # Compute the worker update.
        for group in self.param_groups:
            learning_rate = group["lr"]
            for p in group["params"]:
                p.grad.data *= -learning_rate
        # Send the worker update to the master.
        self._commit()
        # Obtain the new parameterization.
        self._pull()

    def step(self):
        self._step_procedure()

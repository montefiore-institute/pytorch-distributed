import torch
import torch.distributed as dist

from torch.distributed import get_rank, get_world_size

from .optimizer import AsynchronousOptimizer, required
from .utils import is_master



class AEASGD(AsynchronousOptimizer):
    """
    TODO
    """

    def __init__(self, params, lr=required, rho=5.0, workers=None):
        defaults = dict(lr=lr, rho=rho)
        super(AEASGD, self).__init__(params, defaults, workers)

    def _allocate_master_resources(self):
        # Allocate the central variable buffers.
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["buffer"] = torch.zeros_like(p)

    def _master_procedure(self):
        # Receive the rank of the next worker in the queue.
        dist.recv(self.rank.data)
        worker_rank = int(self.rank.item())
        # Acknowledge the worker.
        dist.isend(self.rank.data, dst=worker_rank)
        # Receive the update and store it in the parameter buffer.
        requests = []
        for group in self.param_groups:
            for p in group["params"]:
                parameter_buffer = self.state[p]["buffer"]
                request = dist.irecv(parameter_buffer.data, src=worker_rank)
                requests.append(request)
        # Update central variable.
        for group in self.param_groups:
            for index, p in enumerate(group["params"]):
                parameter_buffer = self.state[p]["buffer"]
                requests[index].wait()
                p.data.add_(parameter_buffer.data)
                dist.isend(p.data, dst=worker_rank)

    def _worker_procedure(self):
        raise NotImplementedError

    def step(self):
        self._step_procedure()

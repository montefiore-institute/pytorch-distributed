import torch
import torch.distributed as dist

from torch.distributed import get_rank, get_world_size

from .optimizer import AsynchronousOptimizer, required
from .utils import is_master



class DOWNPOUR(AsynchronousOptimizer):
    """
    TODO
    """

    def __init__(self, params, lr=required, workers=None):
        defaults = dict(lr=lr)
        super(DOWNPOUR, self).__init__(params, defaults, workers)

    def _allocate_master_resources(self):
        # Allocate the central variable buffers.
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["buffer"] = torch.zeros_like(p)

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
        # Update central variable.
        for group in self.param_groups:
            for index, p in enumerate(group["params"]):
                parameter_buffer = self.state[p]["buffer"]
                requests[index].wait()
                p.data.add_(parameter_buffer.data)
                dist.isend(p.data, dst=worker_rank)

    def _worker_procedure(self):
        # Compute the worker update.
        for group in self.param_groups:
            learning_rate = group["lr"]
            for p in group["params"]:
                p.grad.data *= -learning_rate
        # Send the worker update to the master.
        self._commit()
        # Pull the central variable.
        self._pull()

    def step(self):
        self._step_procedure()

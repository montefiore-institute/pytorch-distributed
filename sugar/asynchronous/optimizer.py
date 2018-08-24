import torch
import torch.distributed as dist

from torch.optim.optimizer import Optimizer, required
from torch.distributed import get_rank, get_world_size

from .utils import is_master



class AsynchronousOptimizer(Optimizer):
    """
    Base class for an asynchronous optimizer.
    """

    def __init__(self, params, defaults=None, workers=None):
        super(AsynchronousOptimizer, self).__init__(params, defaults)
        self.rank = torch.tensor([get_rank()])
        if workers is None:
            self._workers = list(range(1, get_world_size()))
        else:
            self._workers = workers
        if is_master():
            self._allocate_master_resources()
            self._step_procedure = self._master_procedure
        else:
            self._step_procedure = self._worker_procedure
        self._broadcast_parameters()

    def _broadcast_parameters(self):
        for rank in self._workers:
            for group in self.param_groups:
                for p in group["params"]:
                    dist.broadcast(p.data, src=0)

    def _allocate_master_resources(self):
        raise NotImplementedError

    def _master_procedure(self):
        raise NotImplementedError

    def _worker_procedure(self):
        raise NotImplementedError

    def _commit(self):
        # Initiate parameter sharing with the master.
        dist.isend(self.rank, dst=0)
        dist.recv(self.rank, src=0)
        # Send the update to the master
        for group in self.param_groups:
            for p in group["params"]:
                dist.isend(p.grad.data, dst=0)

    def _pull(self):
        # Receive the central variable from the master.
        for group in self.param_groups:
            for p in group["params"]:
                request = dist.irecv(p.data, src=0)
        # Wait for the last receive call.
        request.wait()

    def workers(self):
        return self._workers

    def ready(self):
        if is_master():
            # Acknowledge everyone is done.
            for rank in self._workers:
                self.rank.fill_(rank)
                dist.send(self.rank, dst=rank)
        else:
            dist.recv(self.rank, src=0)
        dist.barrier()

    def step(self):
        raise NotImplementedError

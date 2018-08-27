import socket
import threading
import torch
import torch.distributed as dist

from torch.optim.optimizer import Optimizer, required
from torch.distributed import get_rank, get_world_size

from queue import Queue

from .utils import is_master



class AsynchronousOptimizer(Optimizer):
    """
    Base class for an asynchronous optimizer.
    """

    def __init__(self, params, defaults=None, workers=None):
        super(AsynchronousOptimizer, self).__init__(params, defaults)
        self.rank = torch.tensor([get_rank()])
        self._worker_queue = Queue()
        if workers is None:
            self._workers = list(range(1, get_world_size()))
        else:
            self._workers = workers
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rank_bytes = str(get_rank()).zfill(10).encode()
        self._master_address = (os.environ["MASTER_ADDR"], 5123)
        if is_master():
            self._allocate_master_resources()
            self._step_procedure = self._master_procedure
            self._initialize_socket()
        else:
            self._step_procedure = self._worker_procedure
        self._broadcast_parameters()

    def _initialize_socket(self):
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(self._master_address)

    def _enqueue_master(self):
        self._socket.sendto(self._rank_bytes, self._master_address)

    def _next_worker(self):
        data, address = self._socket.recvfrom(10)
        data = data.decode().split('.')[0]
        rank = int(data)

        return rank

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
        self._enqueue_master()
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
        self._socket.close()
        dist.barrier()

    def step(self):
        raise NotImplementedError

"""
Synchronous (DataParallel) optimization example.
"""

from model import Model

import os
import time
import torch
import torch.distributed as dist
import torch.nn
import torch.nn.parallel
import torch.optim



def main():
    dist.init_process_group(backend="tcp", init_method="env://")
    model = Model()
    train(model)
    dist.barrier()


def train(model):
    batch_size = int(os.environ["BATCH_SIZE"])
    mse = torch.nn.MSELoss()
    model = torch.nn.parallel.DistributedDataParallelCPU(model) # For CPU
    #model = torch.nn.parallel.DistributedDataParallel(model) # For GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    time_start = time.time()
    for i in range(100):
        optimizer.zero_grad()
        x = torch.randn(batch_size, 1, 64, 64)
        y = torch.randn(batch_size, 1)
        loss = mse(model(x), y)
        loss.backward()
        optimizer.step()
    time_duration = time.time() - time_start
    print("Duration", time_duration)


if __name__ == "__main__":
    main()

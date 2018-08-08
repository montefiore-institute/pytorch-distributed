"""
Asynchronous optimization example.
"""

from model import Model

from sugar.asynchronous import DOWNPOUR
from sugar.asynchronous import GEM
from sugar.asynchronous import is_master

import os
import sugar.asynchronous
import time
import torch
import torch.distributed as dist
import torch.nn



def main():
    dist.init_process_group(backend="tcp", init_method="env://")
    model = Model()
    #optimizer = DOWNPOUR(model.parameters(), lr=0.01)
    optimizer = GEM(model.parameters())
    if is_master():
        run_master(model, optimizer)
    else:
        run_worker(model, optimizer)
    optimizer.ready()


def run_worker(model, optimizer):
    batch_size = int(os.environ["BATCH_SIZE"])
    mse = torch.nn.MSELoss()
    for i in range(100):
        optimizer.zero_grad()
        x = torch.randn(batch_size, 1, 64, 64)
        y = torch.randn(batch_size, 1)
        loss = mse(model(x), y)
        loss.backward()
        optimizer.step()


def run_master(model, optimizer):
    num_workers = len(optimizer.workers())
    steps = 100 * num_workers
    time_start = time.time()
    for i in range(steps):
        optimizer.step()
    time_duration = time.time() - time_start
    print("Duration:", time_duration)


if __name__ == "__main__":
    main()

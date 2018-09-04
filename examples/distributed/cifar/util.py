import argparse
import numpy as np
import os
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from sugar.asynchronous import is_master



def parse_arguments():
    parser = argparse.ArgumentParser("CIFAR examples")
    parser.add_argument("--momentum", type=float, default=.9, help="Training momentum term")
    parser.add_argument("--lr", type=float, default=.01, help="Training learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--rank", type=int, default=0, help="Process rank")
    parser.add_argument("--backend", type=str, default="tcp", help="PyTorch distributed backend")
    parser.add_argument("--download", type=bool, nargs='?', const=True, default=False, help="Flag to download the CIFAR dataset")
    parser.add_argument("--downloadm", type=bool, nargs='?', const=True, default=False, help="Only the master can download and process the CIFAR dataset")
    parser.add_argument("--data-path", type=str, default="data/", help="Data path")
    parser.add_argument("--prefetchers", type=int, default=2, help="Number of asynchronous data prefetchers")
    arguments = parser.parse_args()

    return arguments



def download_data(arguments):
    # Check if the dataset needs to be downloaded.
    if arguments.download:
        # Check if only the master needs to download the files.
        if not arguments.downloadm or is_master():
            fetch_training_set()
            fetch_validation_set()
    # Wait for all processes to complete.
    dist.barrier()



def allocate_train_loader(arguments):
    dataset = fetch_training_set()
    loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size,
                                         shuffle=True, num_workers=arguments.prefetchers)

    return loader



def allocate_validation_loader(arguments):
    dataset = fetch_validation_set()
    loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size,
                                         shuffle=True, num_workers=arguments.prefetchers)

    return loader



def fetch_transformations(arguments):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transformations



def fetch_training_set(arguments):
    transform = fetch_transformations()
    data = torchvision.datasets.CIFAR10(root=arguments.data_path, train=True,  download=arguments.download, transform=transform)

    return data


def fetch_validation_set(arguments):
    transform = fetch_transformations()
    data = torchvision.datasets.CIFAR10(root=arguments.data_path, train=False,  download=arguments.download, transform=transform)

    return data


def initialize_distributed_backend(arguments):
    dist.init_process_group(backend=arguments.backend, init_method="env://")

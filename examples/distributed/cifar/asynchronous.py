import torch
import torch.distributed as dist


from model import allocate_model
from sugar.asynchronous import DOWNPOUR
from sugar.asynchronous import GEM
from sugar.asynchronous import is_master
from util import allocate_train_loader
from util import allocate_validation_loader
from util import download_data
from util import initialize_distributed_backend
from util import parse_arguments




def main():
    arguments = parse_arguments()
    initialize_distributed_backend(arguments)
    download_data(arguments)
    model = allocate_model()
    optimizer = GEM(model.parameters(), lr=arguments.lr, momentum=arguments.momentum)
    if is_master():
        master_procedure(arguments, model, optimizer)
    else:
        worker_procedure(arguments, model, optimizer)
    optimizer.ready()



def master_procedure(arguments, model, optimizer):
    num_workers = len(optimizer.workers())



def worker_procedure(arguments, model, optimizer):
    batch_size = arguments.batch_size



if __name__ == "__main__":
    main()

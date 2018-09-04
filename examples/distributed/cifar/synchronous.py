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
from util import num_train_batches
from util import parse_arguments



def main():
    arguments = parse_arguments()
    initialize_distributed_backend(arguments)
    download_data(arguments)
    model = allocate_model()
    model = torch.nn.parallel.DistributedDataParallelCPU(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=arguments.lr, momentum=arguments.momentum)
    worker_procedure(arguments, model, optimizer)



def worker_procedure(arguments, model, optimizer):
    batch_size = arguments.batch_size
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(arguments.epochs):
        loader = allocate_train_loader(arguments)
        for batch_index, (x, y) in enumerate(loader):
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            print(dist.get_rank(), loss.item())
            optimizer.step()
    print("DONE")



def validate(arguments, model):
    loader = allocate_validation_loader(arguments)
    # TODO Implement.
    for batch_index, (x, y) in enumerate(loader):
        pass



if __name__ == "__main__":
    main()

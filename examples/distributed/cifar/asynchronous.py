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
    optimizer = GEM(model.parameters(), lr=arguments.lr, momentum=arguments.momentum)
    if is_master():
        master_procedure(arguments, model, optimizer)
    else:
        worker_procedure(arguments, model, optimizer)
    optimizer.ready()



def master_procedure(arguments, model, optimizer):
    num_workers = len(optimizer.workers())
    batches = num_train_batches(arguments)
    steps = num_workers * batches
    for epoch in range(arguments.epochs):
        print("Starting epoch", (epoch + 1))
        # Start the training steps.
        for step in range(steps):
            optimizer.step()
        print("Epoch completed, starting validation.")
        # Start the validation phase.
        validate(arguments, model)
        print("Validation completed.")



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
            if dist.get_rank() == 1 and batch_index % 10 == 0:
                print(loss.item())
            optimizer.step()



def validate(arguments, model):
    loader = allocate_validation_loader(arguments)
    # TODO Implement.
    for batch_index, (x, y) in enumerate(loader):
        pass



if __name__ == "__main__":
    main()

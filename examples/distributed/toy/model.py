import torch
import torch.nn



class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self._network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            torch.nn.Linear(1296, 1)
        )

    def forward(self, x):
        output = self._network(x)

        return output


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)

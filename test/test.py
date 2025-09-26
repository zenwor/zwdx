import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import os

from zwdx import ZWDX
SERVER_URL = os.environ["SERVER_URL"]
zwdx = ZWDX(SERVER_URL)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CNNPOC(nn.Module):
    def __init__(self):
        super(CNNPOC, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 16 * 26 * 26)
        x = self.fc1(x)
        return x


def get_mnist_loader(rank, world_size):
        transform = transforms.ToTensor()
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        return DataLoader(dataset, batch_size=32, sampler=sampler)


if __name__ == "__main__":
    model = CNNPOC()

    result = zwdx.submit_job(model=model, data_loader_func=get_mnist_loader, parallelism="FSDP", memory_required=12_000_000_000)
    # result = zwdx.submit_job(model=model, data_loader_func=get_mnist_loader, parallelism="DDP", memory_required=12_000_000_000)
    logger.info(f"Job submission result: {result}")

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

import torch.nn.functional as F

def train(model, data_loader, optimizer, eval_func, rank, world_size, epochs=1, reporter=None):
    for epoch in range(epochs):
        model.train()
        epoch_loss, batch_count = 0.0, 0

        for data, target in data_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        final_loss = epoch_loss / batch_count if batch_count > 0 else 0.0

        if reporter:
            metrics = reporter.log_metrics(epoch=epoch, training_loss=final_loss)
            reporter.log(f"Rank {rank} finished training epoch {epoch}")

        if eval_func:
            eval_loss, accuracy = eval_func(model, data_loader, rank, world_size)
            if reporter:
                metrics = reporter.log_metrics(epoch=epoch, eval_loss=eval_loss, accuracy=accuracy)
                reporter.log(f"Rank {rank} evaluated epoch {epoch}")

    return final_loss

def test(model, data_loader, rank, world_size):
    model.eval()
    test_loss, correct, count = 0.0, 0, 0

    with torch.no_grad():
        for data, target in data_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += target.size(0)

    avg_loss = test_loss / count if count > 0 else 0.0
    accuracy = 100.0 * correct / count if count > 0 else 0.0
    return avg_loss, accuracy

if __name__ == "__main__":
    model = CNNPOC()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    result = zwdx.submit_job(
        model=model, 
        data_loader_func=get_mnist_loader, 
        parallelism="DDP", 
        memory_required=12_000_000_000,
        train_func=train,
        eval_func=test,
        optimizer=optimizer,
    )
    logger.info(f"Job submission result: {result}")

    # Retrieve the trained model
    if result["status"] == "complete":
        trained_model = zwdx.get_trained_model(result["job_id"], CNNPOC())
        if trained_model is not None:
            logger.info("Successfully retrieved trained model")
            # Example: Use the trained model for inference
            trained_model.eval()
            sample_data = torch.randn(1, 1, 28, 28)  # Example MNIST input
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            trained_model = trained_model.to(device)
            sample_data = sample_data.to(device)       

            trained_model.eval()
            with torch.no_grad():
                prediction = trained_model(sample_data)
                logger.info(f"Sample prediction: {prediction.argmax(dim=1).item()}")
        else:
            logger.error("Failed to retrieve trained model")
            
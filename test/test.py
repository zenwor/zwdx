import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

from zwdx import ZWDX

SERVER_URL = os.environ["SERVER_URL"]
zwdx = ZWDX(SERVER_URL)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CNNPOC(nn.Module):
    def __init__(self):
        super(CNNPOC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_mnist_loader(rank, world_size, data_path=None):
    """
    Create MNIST DataLoader with distributed sampling support.
    Downloads MNIST if not present.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, DistributedSampler
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download/load MNIST - use /tmp for Docker compatibility
    data_dir = data_path if data_path else "/tmp/mnist_data"
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Create distributed sampler if world_size > 1
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    return loader


def train_mnist(model, data_loader, optimizer, eval_func, rank, world_size, epochs=10, reporter=None):
    """
    Training loop for MNIST with proper loss tracking and reporting.
    """
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.train()
    
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'num_samples'):
        samples_per_rank = data_loader.sampler.num_samples
        if reporter:
            reporter.log(f"Rank {rank}: Processing {samples_per_rank} samples, {len(data_loader)} batches")
    
    first_batch_indices = []
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == 0:
            # Log first 5 target labels to verify different data
            first_labels = target[:5].tolist()
            if reporter:
                reporter.log(f"Rank {rank} first batch labels: {first_labels}")
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        break
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log every 50 batches
            if batch_idx % 50 == 0 and reporter:
                reporter.log_metrics(
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                    accuracy=100. * correct / total
                )
        
        # Epoch summary
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        if reporter:
            reporter.log(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, Total samples={total}")
    
    return avg_loss


def test_mnist(model, data_loader, rank, world_size):
    """
    Evaluation function for MNIST.
    Returns (loss, accuracy).
    """
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

if __name__ == "__main__":
    args = zwdx.parse_args()
    room_token = args.room_token
    
    model = CNNPOC()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Submitting MNIST training job to ZWDX...")
    
    result = zwdx.submit_job(
        model=model,
        data_loader_func=create_mnist_loader,
        train_func=train_mnist,
        eval_func=test_mnist,
        optimizer=optimizer,
        parallelism="FSDP",
        memory_required=12_000_000_000,
        room_token=room_token,
    )

    print("\n" + "="*60)
    print("TRAINING RESULT:")
    print("="*60)
    print(f"Status: {result.get('status')}")
    print(f"Job ID: {result.get('job_id')}")
    if result.get('results'):
        print(f"Final Loss: {result['results'].get('final_loss'):.4f}")
    print("="*60)
    
    # Optionally retrieve the trained model
    if result.get('status') == 'complete':
        trained_model = zwdx.get_trained_model(result['job_id'], CNNPOC())
        if trained_model:
            logger.info("Successfully retrieved trained model weights!")
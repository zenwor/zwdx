import os
import torch
import torch.nn as nn

from zwdx import ZWDX

zwdx = ZWDX(os.environ["SERVER_URL"])
args = zwdx.parse_args()
logger = zwdx.get_logger()

class CNNPOC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def create_loader(rank=0, world_size=1, data_path="/tmp/mnist_data"):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader,DistributedSampler
    
    ds = datasets.MNIST(
        data_path, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    sampler = DistributedSampler(ds, world_size, rank) if world_size > 1 else None
    return DataLoader(ds, batch_size=64, sampler=sampler, shuffle=(sampler is None))

def train(model, loader, optim, eval_func, rank, world_size, epochs=3, reporter=None):
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.train()
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward(); optim.step()
            loss_sum += loss.item(); pred = out.argmax(1)
            correct += pred.eq(target).sum().item(); total += target.size(0)
        if reporter:
            reporter.log_metrics(
                epoch=epoch,
                loss=loss.item(),
                accuracy=100. * correct / total
            )
            reporter.log(f"Epoch {epoch}: Loss={loss_sum/len(loader):.4f}, Acc={100*correct/total:.2f}%")
    return loss_sum / len(loader)

def test(model, loader, rank, world_size):
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.eval(); correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
    return loss_sum/total, 100*correct/total

if __name__ == "__main__":
    logger.info("Submitting MNIST training job...")
    result = zwdx.submit_job(
        model=CNNPOC(),
        data_loader_func=create_loader,
        train_func=train,
        eval_func=test,
        optimizer=torch.optim.Adam(CNNPOC().parameters(), lr=1e-3),
        parallelism="DDP",
        memory_required=12_000_000_000,
        room_token=args.room_token,
    )

    if result.get("status") == "complete":
        logger.info(f"Final loss: {result['results'].get('final_loss'):.4f}")
        trained_model = zwdx.get_trained_model(result['job_id'], CNNPOC())
        if trained_model:
            logger.info("Successfully retrieved trained model weights!")
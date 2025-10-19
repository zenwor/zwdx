<p align="center">
  <img src="logo.png" alt="zwdx" title="zwdx" width="200"/><br>
  distributed. accelerated. done.
</p>

⚠️ This is a personal project. It is not done, as it still only works locally. Will make it work truly remotely hopefully soon. Shouldn't be too problematic.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-%2320232a.svg?logo=react&logoColor=%2361DAFB)](#)
[![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?logo=mongodb&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

**zwdx** is a distributed deep learning platform that enables GPU sharing and collaborative training across multiple machines. Instead of being limited to your local hardware, zwdx allows you to:

- **Distribute training** across multiple GPUs from different contributors
- **Share your idle GPUs** with others who need compute power
- **Submit training jobs** without managing infrastructure
- **Scale seamlessly** from single GPU to multi-node distributed training

Unlike traditional distributed training solutions, zwdx separates compute providers (GPU clients) from compute consumers (job submitters), creating a flexible environment for GPU resources.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
  - [Server](#server)
  - [GPU Client](#gpu-client)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Security Considerations](#security-considerations)
- [Limitations](#limitations)
- [Legal Notice](#legal-notice)
- [License](#license)

## Features

- ✅ **Distributed Training**: Built-in support for DDP / FSDP
- ✅ **GPU Pooling**: Aggregate GPU resources from multiple machines
- ✅ **Room-based Access**: Secure, token-based room system for private GPU sharing
- ✅ **Framework Support**: PyTorch native (additional frameworks coming soon)
- ✅ **Simple API**: Submit jobs with just a few lines of code
- ✅ **Real-time Monitoring**: Track training progress and metrics
- ✅ **Job Management**: Query past jobs
- ✅ **Docker-based**: Consistent environment across all GPU clients

## Architecture

```
┌─────────────────┐
│    zwdx User    │
│   (Your Code)   │
└────────┬────────┘
         │
         │ Submit Job
         ▼
┌─────────────────────────────┐
│    Server                   │
│   ┌─────────────────────┐   │
│   │  Job Pool           │   │
│   ├─────────────────────┤   │
│   │  Room Pool          │   │
│   ├─────────────────────│   │
│   │  Database           │   │
│   └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           │ Distribute 
           │
    ┌──────┴──────┬──────────┐
    ▼             ▼          ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU     │  │ GPU     │  │ GPU     │
│ Client  │  │ Client  │  │ Client  │
│ (Rank 0)│  │ (Rank 1)│  │ (Rank 2)│
└─────────┘  └─────────┘  └─────────┘
```

**How it works:**
1. **Server** manages job pool, GPU client pool, room pool and communication
2. **GPU Clients** join rooms using auth tokens and wait for work
3. **Job Submitters** send training jobs to specific rooms
4. Results are collected and returned to the job submitter

## Prerequisites

### Software Requirements
- **Python**: 3.12.4
- **Docker**: 28.4.0
- **CUDA**: 12.8
- **Operating System**: Linux, Windows (WSL2)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/zenwor/zwdx.git
cd zwdx
```

2. **Install Python dependencies:**
```bash
uv pip install --system -r requirements.txt
```

## Setup

Load environment variables before proceeding:
```bash
cd zwdx/zwdx
source ./setup.sh
```

### Server

The server manages GPU clients, job queues, database, and UI communication. **No GPUs required on the server machine.**

**Start the server:**
```bash
./run_all.sh  # inside zwdx/zwdx/
```

**What runs:**
- Flask server (port 4461)
- Database service (MongoDB, port 5561)
- Web UI (React.js, port 3000)

### GPU Client

GPU clients provide computing power to the server.

**1. Pull NVIDIA base image and build container:**
```bash
docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
docker build -t zwdx_gpu .
```

**2. Launch GPU client:**
```bash
cd zwdx/gpu_client/
./run_gpu_client.sh -rt {ROOM_TOKEN}
```

**Arguments:**
- `-rt, --room_token`: Room authentication token (required)

**Example with specific GPUs:**
```bash
./run_gpu_client.sh -rt my_room_token
```

## Quick Start
The ZWDX interface is designed to be intuitive and Pythonic:

```python
from zwdx import ZWDX

zwdx = ZWDX(server_url="http://localhost:8000")

result = zwdx.submit_job(
    model=YourModel(),                    # PyTorch model instance
    data_loader_func=create_data_loader,  # Function that returns DataLoader
    train_func=train,                     # Training function
    eval_func=eval,                       # Evaluation function
    optimizer=torch.optim.AdamW(...),     # Optimizer instance
    parallelism="DDP",                    # Parallelism strategy
    memory_required=12_000_000_000,       # Minimum GPU memory in bytes
    room_token="your_room_token",         # Room authentication token
    epochs=10,                            # Number of training epochs
)

# Access results
print(result["job_id"])
print(result["results"]["final_loss"])

# Retrieve trained model
trained_model = zwdx.get_trained_model(result["job_id"], YourModel())
```

For a complete MNIST example, see [`test/mnist.py`](./test/mnist.py).

## Configuration

### Environment Variables

Edit the `.env` file or set these environment variables:

```bash
# Server
## Flask
FLASK_HOST="0.0.0.0"
FLASK_PORT=4461
MASTER_ADDR="29500"
LT_SUBDOMAIN="zwdx"
LT_PORT=4461
## MongoDB
MONGODB_PORT=5561
MONGODB_DBPATH="./data/"

# Client
SERVER_URL="http://172.17.0.1:4461"
```
## Monitoring

### Server Logs
```bash
tail -f /var/log/zwdx/server.log
```

### GPU Client Logs
```bash
docker logs -f zwdx_gpu_client_container
```

### Job Metrics
Access real-time metrics via the web UI at `http://localhost:3000` or programmatically:

```python
metrics = zwdx.get_job_metrics(job_id)
print(metrics["gpu_utilization"])
print(metrics["throughput"])  # samples/second
```

### Monitoring Dashboard
The web UI provides:
- Job queue and current jobs
- Training metrics and loss curves
- Historical job performance

## Security Considerations

### Room Tokens
- Room tokens provide isolation between different groups
- Tokens should be treated as sensitive credentials
- Generate strong, random tokens for production use
- Rotate tokens periodically

### Data Privacy
- Training data is sent to GPU client machines
- **Do not use zwdx for sensitive/confidential data unless you trust all GPU providers**
- Model weights are transmitted between clients and server
- Consider using encrypted communication channels for production

## Limitations
- **Framework Support**: PyTorch only (TensorFlow/JAX planned)
- **Parallelism**: DDP / FSDP only (more coming soon)
- **Data Transfer**: Large datasets must be pre-distributed to clients

## Legal Notice

> ⚠️ **NVIDIA Software Usage**
> 
> This project uses NVIDIA software. The base container is proprietary and must be pulled by each user separately:
> ```bash
> docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
> ```
> 
> **Do NOT redistribute the NVIDIA container.** 
> 
> See [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf) for complete terms.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 zenwor

---

**Built with ❤️ by zenwor**
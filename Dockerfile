# Use the official PyTorch container with CUDA from NVIDIA
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Upgrade pip and install any dependencies
RUN pip install --upgrade pip uv

# Copy your requirements and install them
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy your project files
COPY . .

# Expose the port your app uses
EXPOSE 4461

# Default command (adjust to your app)
CMD ["uv", "main:app", "--host", "0.0.0.0", "--port", "4461"]

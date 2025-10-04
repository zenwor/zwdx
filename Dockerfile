FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Set working directory inside container
WORKDIR /app

# Install uv (fast Python package installer)
RUN pip install --upgrade pip uv

# Copy only requirements first (for build caching)
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Now copy the rest of your project into the container
COPY . .

# IMPORTANT: Expose FLASK_PORT env variable
EXPOSE 4461 
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /app
RUN pip install --upgrade pip uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY . .
EXPOSE 4461 
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt torchmetrics

COPY . .
CMD ["python", "scripts/train.py"]
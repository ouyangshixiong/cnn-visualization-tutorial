FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt torchmetrics

COPY . .

# 创建CUDA检查命令别名
RUN echo 'alias check-cuda="python deploy/shared/check_cuda.py"' >> ~/.bashrc

CMD ["python", "scripts/train.py"]
#!/usr/bin/env python3
"""
CUDA环境验证脚本
"""
import torch
import subprocess
import sys

def check_cuda():
    print("=== CUDA环境检查 ===")
    
    # 检查PyTorch CUDA版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu.name} ({gpu.total_memory // 1024**2} MB)")
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n=== nvidia-smi输出 ===")
        print(result.stdout)
    except FileNotFoundError:
        print("未找到nvidia-smi命令")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
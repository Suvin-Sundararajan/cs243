#!/bin/bash
# Run this immediately upon activation of instance. 

# Update the package index
sudo yum update -y

# Install Python3 and pip
sudo yum install python3 python3-pip -y

# Install PyTorch, torchvision, and torchaudio for CUDA 11.3
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install boto3 for AWS SDK for Python
pip3 install boto3
pip3 install tqdm

# Set environment variables for Distributed Data Parallel (DDP)
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Run this immediately upon activation of instance.

# Install Python3 and pip
sudo yum install python3 python3-pip -y

# Update the package index
sudo yum remove cuda -y
sudo yum autoremove -y
sudo rm -rf /usr/local/cuda-9.2
sudo yum update -y

# Add CUDA 11.3 repository
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
sudo yum clean all

# Install CUDA Toolkit for version 11.3
sudo yum -y install cuda-toolkit-11-3

# Install Git
sudo yum install git -y

# Install PyTorch, torchvision, and torchaudio for CUDA 11.3, ensures compatbility
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Installing NCCL: (https://developer.nvidia.com/nccl/nccl-download)
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
sudo yum install libnccl-2.19.3-1+cuda11.0 libnccl-devel-2.19.3-1+cuda11.0 libnccl-static-2.19.3-1+cuda11.0 -y

# Install boto3 for AWS SDK for Python and other utilities
pip3 install boto3
pip3 install tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import tqdm
import os
import time
import boto3
import sys

# Create an S3 client
s3_client = boto3.client('s3')

bucket_name = 'checkpointing'  # Replace with your bucket name
file_name = 'idle_times.txt'
object_key = f'idle_data/{file_name}'

assert(torch.cuda.is_available())
# Setup for Distributed Data Parallel (DDP)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend='nccl')

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the CIFAR-100 training data
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize ResNet50 with pretrained weights
resnet50 = models.resnet50(pretrained=True)
num_classes = 100  # CIFAR-100 has 100 classes
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

# Move model to the device (GPU) and convert it to DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)
resnet50 = DDP(resnet50, device_ids=[dist.get_rank()])

# Optimizer setup
optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)

# Array to store idle times for each GPU
idle_times = []

# Training loop setup
num_epochs = 5  # Just an example, specify the number of epochs you want
if len(sys.argv) > 1:
    try:
        num_epochs = int(sys.argv[1])
    except ValueError:
        print("Usage: python script.py <number_of_epochs>")
        sys.exit(1)

# Wrap trainloader with tqdm on the master process
if dist.get_rank() == 0:
    trainloader = tqdm(trainloader, desc=f"Training Progress", position=0, leave=True)

for epoch in range(num_epochs):
    epoch_idle_time = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = F.cross_entropy(outputs, labels)

        start_sync_time = time.time()
        loss.backward()
        optimizer.step()
        end_sync_time = time.time()

        epoch_idle_time += (end_sync_time - start_sync_time)

        # Update the progress bar with the loss value
        if dist.get_rank() == 0:
            trainloader.set_postfix(loss=loss.item())

    # Accumulate idle time for each epoch
    idle_times.append(epoch_idle_time)

# Gather all idle times from each process to the master process
if dist.get_rank() == 0:
    total_idle_times = [0.0 for _ in range(dist.get_world_size())]
else:
    total_idle_times = None

dist.gather(torch.tensor(idle_times).sum().item(), dst=0, gather_list=total_idle_times if dist.get_rank() == 0 else None)

if dist.get_rank() == 0:
    # Save idle times to a file
    with open(file_name, 'w') as f:
        for idle_time in total_idle_times:
            f.write(f"{idle_time}\n")
    
    # Upload the file to S3
    s3_client.upload_file(file_name, bucket_name, object_key)

    print(f"Uploaded idle times to s3://{bucket_name}/{object_key}")

# Only the master process will print
if dist.get_rank() == 0:
    print("Total Idle Times from All GPUs:", total_idle_times)

dist.destroy_process_group()

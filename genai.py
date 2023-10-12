import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
from torchvision import transforms
import GPUtil
import time

os.environ['MASTER_ADDR'] = 'localhost'  # IP address
os.environ['MASTER_PORT'] = '12355'  #

dist.init_process_group(backend='nccl')


def dataset(): 

    data.load
    data = 5


    return data



# Flags 
DEBUG_FLAG = True  # Turn this flag on for debugging purposes
USE_DISTRIBUTED = True  # True for multi-node different EC2s, False for single-node multi-GPU

CHECKPOINT_PATH = "model_checkpoint.pt"
CHECKPOINT_FREQUENCY = 5  # Save every 5 epochs
MANUAL_CHECKPOINT = False  # Use this flag to manually trigger checkpointing when needed

if USE_DISTRIBUTED:
    os.environ['MASTER_ADDR'] = 'localhost'  # IP address of the master node
    os.environ['MASTER_PORT'] = '12355'  # Any free port of your choice
    dist.init_process_group(backend='nccl')




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(torch.device("cuda:0"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
if USE_DISTRIBUTED:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=(train_sampler is None),
    num_workers=2, pin_memory=True, sampler=train_sampler)

if USE_DISTRIBUTED:
    model = nn.parallel.DistributedDataParallel(model)
else:
    model = nn.DataParallel(model)

def save_checkpoint(epoch, model, optimizer, filename=CHECKPOINT_PATH):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, filename)

for epoch in range(10):
    if USE_DISTRIBUTED:
        train_sampler.set_epoch(epoch)

    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    if DEBUG_FLAG:
        print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, 10, epoch_loss, epoch_acc))

    if (epoch + 1) % CHECKPOINT_FREQUENCY == 0 or (MANUAL_CHECKPOINT and epoch == 9):
        save_checkpoint(epoch, model, optimizer)


def is_gpu_idle(threshold=20):  # Threshold is in percentage
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        if gpu.load * 100 <= threshold:  # If GPU load is below or equal to the threshold
            return True
    return False

def log_gpu_usage(log_file_path="gpu_usage.log", duration=3600, interval=10):
    """
    Log GPU usage to a file.
    
    Parameters:
    - log_file_path (str): Path to the log file.
    - duration (int): Total duration for which GPU usage should be logged (in seconds).
    - interval (int): Time interval between successive logs (in seconds).
    """
    start_time = time.time()
    end_time = start_time + duration
    
    with open(log_file_path, "w") as log_file:
        log_file.write("Time, GPU ID, Load (%), Memory Used (MB)\n")
        
        while time.time() < end_time:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            GPUs = GPUtil.getGPUs()
            
            for gpu in GPUs:
                log_line = f"{timestamp}, {gpu.id}, {gpu.load*100:.2f}, {gpu.memoryUsed:.2f}\n"
                log_file.write(log_line)
                
            time.sleep(interval)

# To start logging GPU usage
log_gpu_usage()


# Command: python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_INSTANCE YOUR_SCRIPT.py
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DataParallel
import torch.optim as optim
from tqdm import tqdm
import time
import datetime
import sys
from gemini_algos import placement_strategy
from gemini_algos import checkpoint_partition
import threading
import copy

lock = threading.Lock()

# CALL THIS FUNCTION WHEN NEEDED AND WANTING TO TEST PARAMETERS

# Variable to control checkpointing location
save_checkpoint_to_cpu = True
checkpoint_frequency = 2  # Checkpoint every 2 epochs
checkpoint_groups = placement_strategy(4,1)
print(checkpoint_groups)

# Ensure CUDA is available
assert(torch.cuda.is_available())
cuda_available = torch.cuda.is_available()

# Count the number of available CUDA devices (GPUs)
num_gpus = torch.cuda.device_count()

# Print the status and number of available GPUs
print(f"CUDA available: {cuda_available}")
print(f"Number of GPUs available: {num_gpus}")

# If CUDA is available, print the names of the GPUs
if cuda_available:
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    print("Available GPUs:")
    for i, name in enumerate(gpu_names):
        print(f"GPU {i}: {name}")

print("Initializing - Correct number of GPUs")

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

# Move model to the device (GPU) and use DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)
resnet50 = DataParallel(resnet50)

# Optimizer setup
optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)

checkpoint = None

def send_checkpoint():
    global checkpoint
    gpu = 1
    while True:
        # suppose for now that we are sending the checkpoint to another GPU every 5 seconds
        # also suppose for now, that we don't need to partition the checkpoint because it is small enough
        time.sleep(5)
        with lock:
            if checkpoint is not None:
                # determine which GPUs to send the checkpoint to from the placement strategy
                group = []
                for i in range(len(checkpoint_groups)):
                    if gpu in checkpoint_groups[i]:
                        group = checkpoint_groups[i]
                        break
                
                for gpu_id in group:
                    if gpu_id != gpu:
                        # this is wrong
                        checkpoint.to(gpu_id)
                checkpoint = None


checkpoint_thread = threading.Thread(target=send_checkpoint)
checkpoint_thread.start()

# Training loop setup
num_epochs = 5  # Just an example, specify the number of epochs you want
if len(sys.argv) > 1:
    try:
        num_epochs = int(sys.argv[1])
    except ValueError:
        print("Usage: python script.py <number_of_epochs>")
        sys.exit(1)

# Function to save the model and optimizer state
# def save_checkpoint(model, optimizer, epoch, file_path='checkpoint.pth'):
#     # If saving to CPU, temporarily move the model to CPU
#     if save_checkpoint_to_cpu:
#         model_state = model.module.cpu().state_dict()
#         # Move model back to GPU after saving state
#         model.to(device)
#     else:
#         model_state = model.module.state_dict()

#     state = {
#         'epoch': epoch,
#         'model_state': model_state,
#         'optimizer_state': optimizer.state_dict(),
#     }
#     torch.save(state, file_path)


# Wrap trainloader with tqdm
trainloader = tqdm(trainloader, desc="Training Progress", position=0, leave=True)

# Variables for periodic updates and idle times
last_update_time = time.time()
update_interval = 60  # seconds
idle_times = []

for epoch in range(num_epochs):
    epoch_idle_time = 0.0
    for i, data in enumerate(trainloader):
        # Periodic update check
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            print(f"Updated: Epoch {epoch + 1}, Batch {i}, Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            last_update_time = current_time

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        start_sync_time = time.time()  # Start idle time measurement
        outputs = resnet50(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        end_sync_time = time.time()  # End idle time measurement
        epoch_idle_time += (end_sync_time - start_sync_time)

        # Update the progress bar
        trainloader.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        trainloader.set_postfix(loss=loss.item())

    idle_times.append(epoch_idle_time)  # Accumulate idle time for each epoch
    print(f"Epoch {epoch + 1} Total Idle Time: {epoch_idle_time} ")

    # Checkpointing logic
    with lock:
        # Create checkpoint
        checkpoint = copy.deepcopy(resnet50.module.state_dict())

    # if (epoch + 1) % checkpoint_frequency == 0:
    #     checkpoint_file = f'checkpoint_epoch_{epoch + 1}.pth'
    #     save_checkpoint(resnet50, optimizer, epoch, file_path=checkpoint_file)
    #     print(f"Checkpoint saved for epoch {epoch + 1}")

# Print total idle times
total_idle_time = sum(idle_times)
print(f"Total Idle Times: {total_idle_time}")

print("Training Complete")

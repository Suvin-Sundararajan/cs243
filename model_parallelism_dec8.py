import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn.functional as F
import sys
import argparse



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.31.10.39'  # Change with ip address
    os.environ['MASTER_PORT'] = '29500'
    print("starting")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Rank successfully connected to the master node at " + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT'])

class SplitResNet(nn.Module):
    def __init__(self, original_model, split_layer):
        super(SplitResNet, self).__init__()
        self.part1 = nn.Sequential(*list(original_model.children())[:split_layer])
        self.part2 = nn.Sequential(*list(original_model.children())[split_layer:])

    def forward_part1(self, x):
        return self.part1(x)

    def forward_part2(self, x):
        return self.part2(x)


def cleanup():
    dist.barrier() 
    dist.destroy_process_group()

def main(rank, world_size, use_big_resnet, num_epochs, checkpoint_frequency, save_checkpoint_to_cpu, checkpoint_groups):
    print(f"Running the basic Model Parallelism example on rank {rank} out of {world_size} processes")
    setup(rank, world_size)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    original_model = models.resnet50(pretrained=True)
    split_layer = 5  # Choose a layer to split the model
    split_model = SplitResNet(original_model, split_layer)
    if rank == 0:
        model_part = split_model.part1
    else:
        model_part = split_model.part2
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the CIFAR-100 training data
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, sampler=train_sampler, pin_memory=False, num_workers=0)
    find_unused_parameters=True
    # Initialize ResNet
    resnet = models.resnet152(pretrained=True) if use_big_resnet else models.resnet50(pretrained=True)
    
    num_classes = 100  # CIFAR-100 has 100 classes
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    resnet.to(device)
    print("ResNet initialized")
    resnet = DDP(resnet, device_ids=[0], output_device=0, find_unused_parameters=True)
    print("DDP initialized")


    # Optimizer setup
    optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        resnet.train()
        train_sampler.set_epoch(epoch)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

        # Checkpointing logic (only by rank 0)
        if rank == 0 and (epoch + 1) % checkpoint_frequency == 0:
            continue

    if rank == 0:
        print("Training Complete")

    cleanup()


params_per_chunk = 2561000
def shard_model(model):
    chunks = []
    current_chunk_params = {}
    items = model.state_dict().items()
    for i in range(len(items)):
        key, value = list(items)[i]
        current_chunk_size = sum(p.numel() for p in current_chunk_params.values())
        if current_chunk_size + value.numel() < params_per_chunk:
            current_chunk_params[key] = value
        else:
            chunks.append(current_chunk_params)
            # Move to the next chunk
            current_chunk_params = {}
            current_chunk_params[key] = value
    return chunks

# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model = torch.nn.DataParallel(model)


# Call the shard() function
print(len(shard_model(model)))

if __name__ == "__main__":
    world_size = 2
    # world_size = torch.cuda.device_count()

    # Check if the expected number of GPUs (4) is available
    # expected_gpu_count = 1
    # assert world_size == expected_gpu_count, f"Expected {expected_gpu_count} GPUs, but found {world_size}"

    use_big_resnet = False
    num_epochs = 5
    checkpoint_frequency = 2
    save_checkpoint_to_cpu = True
    rank = 0

    #TODO: change to call placement_strategy
    checkpoint_groups = [[0, 1], [2, 3]]  

    if len(sys.argv) > 1:
        try:
            num_epochs = int(sys.argv[1])
        except ValueError:
            print("Usage: python script.py <number_of_epochs>")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            rank = int(sys.argv[2])
        except ValueError:
            print("Usage: python script.py <number_of_epochs> <rank>")
            sys.exit(1)
    else:
        print("Usage: python script.py <number_of_epochs> <rank>")
        sys.exit(1)

    os.environ['NCCL_DEBUG'] = 'INFO'
    main(rank, world_size, use_big_resnet, num_epochs, checkpoint_frequency, save_checkpoint_to_cpu, checkpoint_groups)    




# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model = torch.nn.DataParallel(model)



def testing_parameters():
    return 0
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


def shard_model(model, num_shards):
    model_state = model.state_dict()
    total_params = sum(p.numel() for p in model_state.values())
    params_per_shard = total_params // num_shards
    remaining_params = total_params % num_shards  # Handle any remainder

    shards = []
    current_chunk_params = {}
    current_chunk_size = 0

    for key, value in model_state.items():
        if current_chunk_size + value.numel() > params_per_shard and len(shards) < num_shards - 1:
            shards.append(current_chunk_params)
            current_chunk_params = {}
            current_chunk_size = 0

        current_chunk_params[key] = value
        current_chunk_size += value.numel()

    # Add the last chunk (which may be slightly larger due to the remainder)
    if current_chunk_params:
        shards.append(current_chunk_params)

    # Adjust the last shard size to include any remaining parameters
    if remaining_params > 0 and shards:
        last_shard_key, last_shard_value = list(shards[-1].items())[-1]
        shards[-1][last_shard_key] = last_shard_value[:remaining_params]

    return shards

def save_specific_shards(shards, shard_indices, saved_shards):
    for i in shard_indices:
        
        filename = f"model_shard_{i}.pth"
        torch.save(shards[i], filename)
        saved_shards[i] = True  # Mark the shard as saved
        print(f"Shard {i} saved as {filename}")
       




# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model = torch.nn.DataParallel(model)


# EXAMPLE



num_shards = 5  # desired number of shards
shards = shard_model(model, num_shards)
saved_shards = [False] * num_shards

# Save specific shards


# Later, you can save the remaining shards


# Check saved shards



# Call the shard() function


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
            if epoch == 0 and rank == 0 :
                print("SAVING ON MAIN)")
                save_specific_shards(shards, [1, 2, 3], saved_shards)
                print("Shards saved:", saved_shards)
            if epoch == 1 and rank == 1:
                print("SAVING")
                save_specific_shards(shards, [4], saved_shards)
                print("Shards saved:", saved_shards)
            if rank == 0 and i % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

        # Checkpointing logic (only by rank 0)
        if rank == 0 and (epoch + 1) % checkpoint_frequency == 0:
            continue

    if rank == 0:
        print("Training Complete")

    cleanup()


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
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import time
import datetime
import sys
import copy
import threading
from gemini_algos import placement_strategy
from gemini_algos import checkpoint_partition


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'IP'  # Change with ip address
    os.environ['MASTER_PORT'] = '12345'
    print("starting")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Rank successfully connected to the master node at " + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT'])

def cleanup():
    dist.barrier()  # Ensure all processes reach this point before proceeding to the next line
    dist.destroy_process_group()

def main(rank, world_size, use_big_resnet, num_epochs, checkpoint_frequency, save_checkpoint_to_cpu, checkpoint_groups):
    print(f"Running the basic DDP example on rank {rank} out of {world_size} processes")
    setup(rank, world_size)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

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
            # make a copy of resnet.module and store it in cpu
            # model_copy = copy.deepcopy(resnet.module)
            # model_copy.cpu()

            # # make a copy of resnet.module and move it to the other gpus
            # group = []
            # for i in range(len(checkpoint_groups)):
            #     if rank in checkpoint_groups[i]:
            #         group = checkpoint_groups[i]
            #         break
            # for gpu_id in group:
            #     model_copy_gpu = copy.deepcopy(resnet.module)
            #     if gpu_id != rank:
            #         model_copy_gpu.cuda(gpu_id)


            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'model_state': resnet.module.state_dict(),
            #     'optimizer_state': optimizer.state_dict(),
            # }

            # torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
            # print(f"Checkpoint saved for epoch {epoch + 1}")

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

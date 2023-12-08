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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier() 
    dist.destroy_process_group()

class ModelParallelResNet50(nn.Module):
    def __init__(self, split_size, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__()
        self.split_size = split_size

        # First part of ResNet50
        self.model_part1 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:self.split_size]).cuda(0)
        
        # Second part of ResNet50
        self.model_part2 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[self.split_size:]).cuda(1)

    def forward(self, x):
        inputs1 = x.to('cuda:0')
        inputs2 = self.model_part1(inputs1).to('cuda:1')
        outputs = self.model_part2(inputs2)
        return outputs

def train(rank, world_size, epochs):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=train_sampler)

    split_size = 5
    model = ModelParallelResNet50(split_size=split_size).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        ddp_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Rank of the process')
    args = parser.parse_args()

    world_size = 2
    epochs = 5

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    train(args.rank, world_size, epochs)

if __name__ == "__main__":
    main()

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
from multiprocessing import Process, Manager, set_start_method, Queue
from gemini_algos import placement_strategy
from gemini_algos import checkpoint_partition


params_per_chunk = 2561000
# number of seconds to wait before sending the next chunk
send_chunk_frequency = 5

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '18.220.20.182'  # Change with ip address
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Rank successfully connected to the master node at " + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT'])

def cleanup():
    dist.destroy_process_group()


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
        
    

def send_chunk(rank, model_chunks):
    print('Chunk sending process started')
    setup(rank+2, 4)
    print('Chunk sending process connected to the master node')

   
    while True:
        if model_chunks.qsize() > 0:
            print('Sending chunk to the other machine')
            # Get the first chunk
            chunk = model_chunks.get()

            # TODO: Send to the other machine
            target_rank = 1 if rank == 0 else 0
            chunk_tensor = torch.cat([value.flatten() for value in chunk.values()])
            dist.send(tensor=chunk_tensor, dst=target_rank)
            print('Chunk sent to rank ' + str(target_rank))
        else:
            print('No chunks to send. Waiting for ' + str(send_chunk_frequency) + ' seconds')
            
        # Wait for 5 seconds
        time.sleep(send_chunk_frequency)

def main(rank, world_size, use_big_resnet, num_epochs, checkpoint_frequency, save_checkpoint_to_cpu, model_chunks):
    print(f"Running basic DDP example on rank {rank} out of {world_size} processes")
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, sampler=train_sampler)

    # Initialize ResNet
    resnet = models.resnet152(pretrained=True) if use_big_resnet else models.resnet50(pretrained=True)
    num_classes = 100  # CIFAR-100 has 100 classes
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    resnet.to(device)
    print("ResNet initialized")
    resnet = DDP(resnet)
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

            if i % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

        # Checkpointing logic (only by rank 0)
        chunks = shard_model(resnet)
        for chunk in chunks:
            model_chunks.put(chunk)
        print('Model has been sharded into ' + str(len(chunks)) + ' chunks')
        model_copy = copy.deepcopy(resnet.module)
        model_copy.cpu()
        print('Model has been saved to local CPU')


    if rank == 0:
        print("Training Complete")

    cleanup()

if __name__ == "__main__":
    set_start_method('spawn')
    world_size = 4
    # world_size = torch.cuda.device_count()

    # Check if the expected number of GPUs (4) is available
    # expected_gpu_count = 1
    # assert world_size == expected_gpu_count, f"Expected {expected_gpu_count} GPUs, but found {world_size}"

    use_big_resnet = False
    num_epochs = 5
    checkpoint_frequency = 2
    save_checkpoint_to_cpu = True
    rank = 0

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

    with Manager() as manager:
        # Create a shared list to store the chunks
        model_chunks = Queue()
        # Create a Process for sending chunks
        chunk_process = Process(target=send_chunk, args=(rank, model_chunks))
        chunk_process.start()

        main(rank, world_size, use_big_resnet, num_epochs, checkpoint_frequency, save_checkpoint_to_cpu, model_chunks)   
        
        # Wait for the chunk sending process to finish
        chunk_process.join() 
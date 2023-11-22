import torch

# Check if CUDA is available
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

import torch
import torchvision.models as models

params_per_chunk = 2561000
def shard_model(model):
    chunks = []
    current_chunk = 0
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
            current_chunk += 1
            current_chunk_params = {}
            current_chunk_params[key] = value
    return chunks

# Load the pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model = torch.nn.DataParallel(model)

# Call the shard() function
print(len(shard_model(model)))

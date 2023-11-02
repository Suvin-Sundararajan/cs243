
# pip install transformers
# pip install deepspeed


import math
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import deepspeed
from deepspeed import DeepSpeedEngine
# this is used for the ZeRO-3 Setting

from transformers import GPT2LMHeadModel, GPT2Config

# Define GPT-2 config
config = GPT2Config(
    vocab_size=50265,
    n_positions=1024,
    n_ctx=1024,
    n_embd=1600,
    n_layer=48,
    n_head=25,
    # ... other configurations
)

# Initialize GPT-2 model
model = GPT2LMHeadModel(config=config)

from deepspeed import DeepSpeedEngine

# Define DeepSpeed config as a Python dictionary
deepspeed_config = {
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": False,
        "contiguous_memory_optimization": False,
        "profile": False
    }
}


# Initialize DeepSpeed
model, _, _, _ = DeepSpeedEngine.initialize(model, deepspeed_config=deepspeed_config)





def placement_strategy(N, m):
    # N is the number of GPU machines and m is the number of checkpoint replicas
    G = []
    g = N // m

    for i in range(g):
        G_i = []
        for j in range(1, m+1):
            G_i.append(m * i + j)
        G.append(G_i)

    strategy = "group"
    if N % m != 0:
        strategy = "mixed"
        # Add remaining machines to the last group
        for j in range(g * m + 1, N + 1):
            G[-1].append(j)

    return G, strategy

# Example usage:
print(placement_strategy(5, 2)) 


# Or, implement it yourself using factorials
def ncr(n, k):
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))



def failure_recoverability_prob(N,m,k): 
    # Collorary 1
    if k < m: 
        return 1
    else: 
        return max(0,1- (N *  ncr(N-m, k-m))/(m* ncr(N,k)))
    
# Example usage
print(failure_recoverability_prob(16,2,2))


def checkpoint_partition(T, C, m, p, R, B, mu, f):
    T.append(float('inf'))
    partitions = []
    cpkt_id = 0
    remain_size = C
    
    for t in T:
        remain_span = mu * t
        while remain_span > 0:
            if remain_span >= f(R/p):
                size = R/p
            else:
                size = max(0, remain_span - B)  # The algorithm provides '-αB', but α is not defined. Assuming it is 1 for the subtraction.
            size = min(remain_size, size)
            if size > 0:
                remain_size -= size
                remain_span -= f(size)
                partitions.append(size)
            if remain_size == 0:
                if cpkt_id < m - 1:
                    cpkt_id += 1
                    remain_size = C
                else:
                    return partitions
    return partitions



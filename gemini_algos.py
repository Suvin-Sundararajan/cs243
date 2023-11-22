import math

# Functions: 
# 1. placement_strategy(N, m)
# 2. checkpoint_partition(T, C, m, p, R, a, B, mu, f)
# 3. failure_recoverability_prob(16,2,2)



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

print(placement_strategy(5, 2)) 

def checkpoint_partition(T, C, m, p, R, a, B, mu, f):
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
                size = max(0, remain_span - a * B)  # The algorithm provides '-αB', but α is not defined. Assuming it is 1 for the subtraction.
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


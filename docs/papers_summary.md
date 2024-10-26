# Paper Summary

- Key points summarized for papers centered around distributed training

| Paper                             | Parallelism Scheme   | Overlap Scheme |
| :-------------------------------- | -------------------- | -------------- |
| Megatron-LM                       |                      |                |
| Reducing Activation Recomputation |                      |                |
| DeepSpeed Ulysses                 | Sequence Parallelism | No overlap     |
| Centauri                          |                      |                |
| Ring Attention                    |                      |                |

## DeepSpeed Ulysses

![DeepSpeed Ulysses Design](../figs/deepspeed-ulysses-design.png)

### Workflow

- Partition in _sequence_ dimension of the input embedding matrix and get $Q, K , V$ matrices with matmul
- Before attention module, _alltoall_ communication
  - Partition from sequence dimension to embedding dimension for each device (necessity comes from the multi-head attention design)
- After attention module, _alltoall_ communication, and then MLP matmul, layernorm etc.
  - Partition from embedding dimension to sequence dimension for each device 

### Communication Analysis

- $N$ - sequence length

- $h$ - hidden size
- $P$ - #GPU

- This work
  - before attention communicaiton: $3Nh/P$ per link
  - after attention communication: $Nh/P$ per link
  - total: $4Nh/P = O(N/P)$
  - Pros: remain constant when $N$ and $P$ are increased proportionally --> support longer sequence with more GPUs

- Megatron-LM
  - $O(N)$
  - two all-gather with message volume of $Nh$
  - two reduce-scatter with the volume of $Nh$
  - Their cost of size $M$ remains $M$ when $P > 1$ instead of $M/P$

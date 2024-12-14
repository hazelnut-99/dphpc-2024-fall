"""
Validate if two Nanotron model attention outputs are equal.
Command:
    torchrun --nproc_per_node=1 validate.py --checkpoint_path_1=nanotron-1-path --checkpoint_path_2=nanotron-2-path
"""

from argparse import ArgumentParser
from log import *
import torch
from nanotron.parallel.sequence_parallel.ring_flash_attn.utils import zigzag_split

log_level = 0

def compare_tensors(tensor1, tensor2, torlerance=1/128):
    diff = torch.abs(tensor1 - tensor2)
    if log_level == 1:
        log_info(f'Min diff: {torch.min(diff)}')
        log_info(f'Mean diff: {torch.mean(diff)}')
        log_info(f'Max diff: {torch.max(diff)}')
    if torch.any(diff > torlerance):
        return False
    return True

if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two Nanotron model attention outputs to check if they are equal")
    parser.add_argument("--ranks", type=int, default=2, help="Rank of the process in SP model")
    parser.add_argument("--log-level", type=str, default="SUCC", help="Logging level")
    args = parser.parse_args()

    if args.log_level == "INFO":
        log_level = 1

    # Load the attention outputs of vanilla Nanotron model
    vanilla_path = f"./checkpoints/vanilla_output.pt"
    vanilla_output = torch.load(vanilla_path, map_location=torch.device("cuda"), weights_only=True)["hidden_states"]

    # Load the attention outputs of Nanotron model with SP
    for r in range(args.ranks):
        sp_path = f"./checkpoints/sp_output_r{r}.pt"
        sp_output = torch.load(sp_path, map_location=torch.device("cuda"), weights_only=True)["hidden_states"]
        vaniall_local_output = zigzag_split(r, args.ranks, vanilla_output, dim=0)[0]
        if log_level == 1:
            log_info(f'Rank {r} SP output: {sp_output}')
            log_info(f'Rank {r} SP output shape: {sp_output.shape}')
            log_info(f'Rank {r} Vanilla local output: {vaniall_local_output}')
            log_info(f'Rank {r} Vanilla local output shape: {vaniall_local_output.shape}')
        if not compare_tensors(sp_output, vaniall_local_output):
            log_error(f"Attention outputs of vanilla Nanotron and Nanotron with SP are not equal for rank {r}")
    
    log_success("All attention outputs are equal")
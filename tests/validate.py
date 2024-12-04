"""
Validate if two Nanotron models are equal.
Command:
    torchrun --nproc_per_node=1 validate.py --checkpoint_path_1=nanotron-1-path --checkpoint_path_2=nanotron-2-path
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from convert_weights import load_nanotron_model
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models.llama import LlamaForTraining

from log import *


def load_model(
    checkpoint_path: Path
) -> LlamaForTraining:
    """Load a Nanotron model from a checkpoint."""
    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = NanotronLlamaConfig(**attrs)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
    )

    return nanotron_model


def compare_models(
    nanotron_model_1: LlamaForTraining, 
    nanotron_model_2: LlamaForTraining, 
    tolerance: float = 1e-5
) -> bool:
    """Compares two nanotron models to check if they are equal within a certain tolerance."""
    nanotron_model_1_state_dict = nanotron_model_1.state_dict()
    nanotron_model_2_state_dict = nanotron_model_2.state_dict()
    for key in nanotron_model_1_state_dict.keys():
        if key in nanotron_model_2_state_dict:
            diff = torch.abs(nanotron_model_1_state_dict[key] - nanotron_model_2_state_dict[key])
            if torch.any(diff > tolerance):
                log_error(f"Model weights differ for key {key}.")
                log_error(f"Max difference: {torch.max(diff)}")
                return False
        else:
            log_error(f"Key {key} not found in model 2.")
            return False
        log_info(f"Model weights are equal for key {key}.")
    log_success("Models are equal.")
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two Nanotron models to check if they are equal")
    parser.add_argument("--checkpoint_1_path", type=Path, default="checkpoints/vanilla/100", help="Path to the first checkpoint")
    parser.add_argument("--checkpoint_2_path", type=Path, default="checkpoints/sp/100", help="Path to the second checkpoint")
    args = parser.parse_args()

    # Load Nanotron models.
    nanotron_model_1 = load_model(args.checkpoint_1_path)
    nanotron_model_2 = load_model(args.checkpoint_2_path)

    # Compare models.
    compare_models(nanotron_model_1, nanotron_model_2)

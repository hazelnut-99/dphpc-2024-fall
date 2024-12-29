import yaml
import subprocess
import os
from datetime import datetime
import json

train_configs = [
    {'gpus': 1, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 2, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 2, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 4, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 4, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 4, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 2, 'num_attention_heads': 4},
    {'gpus': 1, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 2, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 2, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 2, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 4, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 4, 'sp_ulysses': 1, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 4, 'num_attention_heads': 4},
    {'gpus': 4, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 2, 'num_attention_heads': 4},

    {'gpus': 1, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 2, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 2, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 4, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 4, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 4, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 2, 'num_attention_heads': 2},
    {'gpus': 1, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 2, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 2, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 2, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 4, 'sp_ring': 1, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 4, 'sp_ulysses': 1, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 1, 'sp_ulysses': 2, 'num_attention_heads': 2},
    {'gpus': 4, 'dp': 1, 'sp_ring': 2, 'sp_ulysses': 2, 'num_attention_heads': 2},
]


def render_yaml_content(conf):
    with open('base_template.yaml', 'r') as file:
        data = yaml.safe_load(file)

    data['parallelism']['dp'] = conf['dp']
    data['parallelism']['sp_ring'] = conf['sp_ring']
    data['parallelism']['sp_ulysses'] = conf['sp_ulysses']
    data['model']['model_config']['num_attention_heads'] = conf['num_attention_heads']
    data['tokens']['sequence_length'] = conf['sequence_length']

    return data


def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def execute_shell_command(cmd):
    print(f"executing command: {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    return p.returncode


def run_one_config(conf, config_path):
    # create a directory using conf_id
    print("Training using config: " + ", ".join(f"{key}={value}" for key, value in conf.items()))
    top_directory = f"output/{config_path}"
    create_directory(top_directory)
    # generate the yaml file
    with open(f'{top_directory}/conf.yaml', 'w') as file:
        yaml.dump(render_yaml_content(conf), file)

    # submit job
    command = f"sbatch submit.sh --output={top_directory}/output.o --error={top_directory}/error.e --gpus-per-task={conf['gpus']} {conf['gpus']} {top_directory}"
    rc = execute_shell_command(command)

    # collect result dump to file
    outcome = dict(conf)
    outcome['result'] = 'success' if rc == 0 else 'fail'

    with open(f'{top_directory}/outcome.json', 'w') as json_file:
        json.dump(outcome, json_file, indent=4)
    return rc == 0


def run():
    prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"working directory prefix: {prefix}")
    index = 0
    for conf in train_configs:
        seq_len = 256
        while True:
            conf['sequence_length'] = seq_len
            config_path = f"{prefix}/{index}"
            success = run_one_config(conf, config_path)
            seq_len <<= 2
            index += 1
            if not success or seq_len > 16384:
                break


run()








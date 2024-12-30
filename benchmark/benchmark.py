import yaml
import subprocess
import os
from datetime import datetime
import json
import time
import shutil
import uuid

resource_conf = {
    "time": "04:00:00",
    "nodelist": "ault[25]",
    "ntasks-per-node": 1,
    "mem": "200G",
    "account": "g34"
}


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
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


def execute_shell_command(cmd):
    print(f"executing command: {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (resultData, resultErr) = p.communicate()
    print(resultData)
    print(resultErr)
    return p.returncode


def prepare_one_config(conf, config_path):
    print("Generating config: " + ", ".join(f"{key}={value}" for key, value in conf.items()))
    top_directory = f"output/{config_path}"
    create_directory(top_directory)

    # generate the yaml file
    with open(f'{top_directory}/conf.yaml', 'w') as file:
        yaml.dump(render_yaml_content(conf), file)

    with open(f'{top_directory}/gpus', 'w') as file:
        file.write(str(conf['gpus']))

    with open(f'{top_directory}/parameters.json', 'w') as json_file:
        json.dump(conf, json_file, indent=4)


def prepare_configs():
    prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"working directory prefix: {prefix}")
    for conf in train_configs:
        seq_len = 256
        while seq_len <= 65536:
            conf['sequence_length'] = seq_len
            generated_uuid = str(uuid.uuid4())
            config_path = f"{prefix}/{generated_uuid}"
            prepare_one_config(conf, config_path)
            seq_len <<= 2
    return prefix


prefix = prepare_configs()
command = f"sbatch run.sh ./output/{prefix}"
execute_shell_command(command)










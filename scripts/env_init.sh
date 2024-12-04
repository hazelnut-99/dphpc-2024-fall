#!/bin/bash

module load cuda/11.8.0 gcc/10.2.0 ninja/1.10.2

conda create -n nanotron-sp python=3.10 -y
conda activate nanotron-sp

echo 'alias env_init="conda activate nanotron-sp; module load cuda/11.8.0 gcc/10.2.0 ninja/1.10.2"' >> ~/.bashrc
source ~/.bashrc
init

pip install --upgrade pip
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install datasets transformers
pip install ninja
pip install flash-attn==2.6.3 --no-build-isolation
pip install colorama

cd ../src/nanotron-vanilla
pip install -e .
cd -

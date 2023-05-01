#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/spike_ddp/launcher.py \
      --data $HOME2/scratch/spike_contrastive/dy016/ \
      --workers 32 \
      --epochs 800 \
      --batch-size 128 \
      --out_dim 5 \
      --proj_dim 5 \
      --optimizer adam \
      --learning-rate 0.001 \
      --checkpoint-dir $HOME2/scratch/spike_ddp/saved_models/ \
      --log-dir $HOME2/scratch/spike_ddp/logs/ \
      --ngpus-per-node 4 \
      --nodes 2 \
      --exp mc_gpt_out5_proj5_bs128_embd32_block121_ep800_lr0.001 \
      --block_size 121 \
      --n_embd 32 
      --multi_chan
echo "Run completed at:- "
date


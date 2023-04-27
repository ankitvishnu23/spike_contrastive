#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.001
do
python \
      $HOME2/scratch/spike_contrastive/launcher.py \
      --data $HOME2/scratch/spike_contrastive/dy016/ \
      --epochs 800 \
      --batch-size 128 \
      --out_dim 5 \
      --proj_dim 5 \
      --lr ${lr} \
      --arch attention \
      --checkpoint-dir $HOME2/scratch/spike_contrastive/saved_models/ \
      --log-dir $HOME2/scratch/spike_contrastive/logs/ \
      --nodes 2 \
      --ngpus-per-node 4 \
      --exp mc_gpt_out5proj5_bs128_embd32_block121_ep800_lr${lr} \
      --fp16 \
      --use_gpt \
      --block_size 121 \
      --n_embd 32 
done
echo "Run completed at:- "
date
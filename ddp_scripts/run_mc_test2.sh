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
      --batch-size 120 \
      --out_dim 5 \
      --proj_dim 5 \
      --lr ${lr} \
      --arch attention \
      --multi_chan \
      --checkpoint-dir $HOME2/scratch/spike_contrastive/saved_models/ \
      --log-dir $HOME2/scratch/spike_contrastive/logs/ \
      --nodes 3 \
      --ngpus-per-node 4 \
      --exp mc_test2 \
      --fp16 \
      --use_gpt \
      --block_size 1331 \
      --n_embd 32 
done
echo "Run completed at:- "
date
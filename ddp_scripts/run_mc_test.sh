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
      $HOME2/scratch/spike_contrastive/launcher.py \
      --data $HOME2/scratch/spike_contrastive/dy016/ \
      --epochs 800 \
      --batch-size 64 \
      --out_dim 5 \
      --proj_dim 5 \
      --lr 0.0001 \
      --arch attention \
      --multi_chan \
      --checkpoint-dir $HOME2/scratch/spike_contrastive/saved_models/ \
      --log-dir $HOME2/scratch/spike_contrastive/logs/ \
      --nodes 1 \
      --ngpus-per-node 2 \
      --exp mc_test

echo "Run completed at:- "
date
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
      $HOME2/scratch/spike_ddp/launcher.py \
      --data $HOME2/scratch/spike_data/single_dy016_random_neurons_04_28_2023 \
      --workers 32 \
      --epochs 800 \
      --batch-size 512 \
      --out_dim 5 \
      --proj_dim 5 \
      --optimizer adam \
      --learning-rate ${lr} \
      --checkpoint-dir $HOME2/scratch/spike_ddp/saved_models/ \
      --log-dir $HOME2/scratch/spike_ddp/logs/ \
      --ngpus-per-node 4 \
      --nodes 2 \
      --exp 0502_single_block121_bs512_lr${lr}_8gpu \
      --block_size 121 \
      --n_embd 64 \
      --pos_enc conseq \
      --is_causal \
      --knn-freq 1
done
echo "Run completed at:- "
date


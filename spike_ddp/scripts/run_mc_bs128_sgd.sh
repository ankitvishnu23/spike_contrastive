#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.01 0.1 1.0
do
python \
      $HOME2/scratch/spike_ddp/launcher.py \
      --data $HOME2/scratch/spike_contrastive/dy016 \
      --workers 32 \
      --epochs 800 \
      --batch-size 132 \
      --out_dim 5 \
      --proj_dim 5 \
      --optimizer sgd \
      --learning-rate ${lr} \
      --checkpoint-dir $HOME2/scratch/spike_ddp/saved_models/ \
      --log-dir $HOME2/scratch/spike_ddp/logs/ \
      --ngpus-per-node 4 \
      --nodes 3 \
      --exp mc_gpt_posseq_causal_nembd32_block1331_bs132_sgd_lr${lr} \
      --block_size 1331 \
      --n_embd 32 \
      --multi_chan \
      --pos_enc seq_11times \
      --is_causal 
done
echo "Run completed at:- "
date


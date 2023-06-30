#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.0001
do
python \
      $HOME2/scratch/spike_contrastive/launcher.py \
      --data $HOME2/scratch/spike_data/multi_dy016_random_neurons_05_16_2023 \
      --workers 32 \
      --epochs 800 \
      --batch-size 512 \
      --out_dim 5 \
      --proj_dim 5 \
      --optimizer adam \
      --learning-rate ${lr} \
      --checkpoint-dir $HOME2/scratch/spike_contrastive/saved_models/ \
      --log-dir $HOME2/scratch/spike_contrastive/logs/ \
      --ngpus-per-node 4 \
      --nodes 4 \
      --exp 0517_outdim5proj5_mc_gpt_conseq_causal_nembd64_block363_bs512_extra1_lr${lr}_knn10_addtrain \
      --block_size 363 \
      --n_embd 64 \
      --multi_chan \
      --pos_enc conseq \
      --is_causal \
      --num_extra_chans 1 \
      --knn-freq 10 \
      --add_train 
done
echo "Run completed at:- "
date


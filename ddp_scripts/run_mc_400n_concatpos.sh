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
      --data $HOME2/scratch/spike_data/multi_400_random_neurons_5chan_05_19_2023 \
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
      --nodes 8 \
      --exp 0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr${lr}_concatpos \
      --block_size 610 \
      --n_embd 64 \
      --multi_chan \
      --pos_enc conseq \
      --is_causal \
      --num_extra_chans 2 \
      --knn-freq 50 \
      --add_train \
      --num_classes 400 \
      --use_chan_pos \
      --concat_pos 


done
echo "Run completed at:- "
date


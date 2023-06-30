#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=tim
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.001 0.0005 0.0001
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
      --nodes 1 \
      --exp 0612_FC_outdim5proj5_mc_block605_bs512_extra2_lr${lr}_knn10_addtrain \
      --block_size 605 \
      --multi_chan \
      --num_extra_chans 2 \
      --knn-freq 50 \
      --add_train \
      --num_classes 400 \
      --use_fc

done
echo "Run completed at:- "
date





#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.003 0.03
do
python \
      $HOME2/scratch/ssl/imagenet/simclr/launcher_linear.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/checkpoints/moco_v2_800ep_pretrain.pth \
      --checkpoint-dir $HOME2/scratch/ssl/imagenet/simclr/saved_models/ \
      --log-dir $HOME2/scratch/ssl/imagenet/simclr/logs/ \
      --nodes 1 \
      --exp ep_MoCo_ft_lr${lr}_bs1024 \
      --lr-backbone $lr \
      --lr-classifier $lr \
      --weights finetune \
      --batch-size 1024 \
      --ngpus-per-node 4
done
echo "Run completed at:- "
date

#!/bin/bash

script=$1
shift
for lr in 0.001 
do
python wsc_launcher.py --dist-url "tcp://$MASTER_HOSTNAME:10596" --dist-backend 'nccl' --multiprocessing-distributed \
                --world-size $NODES --rank ${PMIX_RANK} --data /gpfs/wscgpfs02/shivsr/cloh/spike_data/multi_dy016_random_neurons_04_28_2023 --workers 32 \
                --batch-size 120 --epochs 800 --learning-rate ${lr} --checkpoint-dir /gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/saved_models \
                --log-dir /gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/logs --exp 0510_mc_conseq_causal_n64_b1331_bs120_extra2_lr${lr}_poschan --n_embd 64 --block_size 1331 --multi_chan --pos_enc conseq --is_causal --num_extra_chans 2 --knn-freq 10 --use_chan_pos $script "$@"
done

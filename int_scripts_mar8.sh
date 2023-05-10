for lr in 0.001 0.0001 0.0003
do
for fcd in 2 4
do
python run.py --submit --arg_str="--arch=custom_encoder2 --out_dim=256 --proj_dim=64 --batch-size=512 --lr=${lr} --fc_depth=${fcd} "
done
done

for lr in 0.001 0.0001 0.00001 0.01
do
python run.py --submit --arg_str="--out_dim=20 --proj_dim=10 --batch-size=512 --lr=${lr} "
done


for lr in 0.001 0.0001 0.01
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=500 "
done

for lr in 0.001 0.1 0.01 1 0.3
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=500 --optimizer=sgd "
done

for lr in 0.001 0.0001 0.01
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=500 --fc_depth=4 "
done

for lr in 0.001 0.1 0.01 1 0.3
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=500 --optimizer=sgd --fc_depth=4 "
done


# for lr in 0.001 0.1 0.01 1 0.3
# do
# for wd in 1e-5 1e-4 1e-3
# do
# python finetune.py --submit --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --optimizer=sgd --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500 "
# python finetune.py --submit --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --optimizer=sgd --pt_ckpt=out_dim5proj_dim5batch-size512lr0.0001archattentionepochs500 "
# python finetune.py --submit --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --optimizer=sgd --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500fc_depth4 "
# done
# done

for lr in 0.001 0.0001 0.01 0.0005 0.00005
do
for wd in 1e-5 1e-4 1e-3
do
python finetune.py --submit --add_prefix=0329 --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500 "
python finetune.py --submit --add_prefix=0329 --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --pt_ckpt=out_dim5proj_dim5batch-size512lr0.0001archattentionepochs500 "
python finetune.py --submit --add_prefix=0329 --arg_str="--cls_head=mlp3 --lr=${lr} --wd=${wd} --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500fc_depth4 "
done
done

# out_dim5proj_dim5batch-size512lr0.001archattentionepochs500
# out_dim5proj_dim5batch-size512lr0.0001archattentionepochs500
# out_dim5proj_dim5batch-size512lr0.001archattentionepochs500fc_depth4
# out_dim5proj_dim5batch-size512lr0.01archattentionepochs500optimizersgdfc_depth4_checkpoint_0500.pth.tar

python finetune.py --cls_head=mlp3 --lr=0.01 --wd=1e-3 --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500


for lr in 0.001 0.0001 0.01 0.0005 0.00005
do
for wd in 1e-5 1e-4 1e-3
do
python finetune.py --submit --add_prefix=0329 --arg_str="--cls_head=mlp2 --lr=${lr} --wd=${wd} --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500 "
python finetune.py --submit --add_prefix=0329 --arg_str="--cls_head=linear --lr=${lr} --wd=${wd} --pt_ckpt=out_dim5proj_dim5batch-size512lr0.001archattentionepochs500 "
done
done

for lr in 0.001 
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=1000 "
done


for lr in 0.001 0.0001 0.01 0.003 0.0003
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=800 --multi_chan "
done

python run.py --out_dim=5 --proj_dim=5 --batch-size=8 --lr=0.01 --arch=attention --epochs=800 --multi_chan 

python run.py --out_dim=5 --proj_dim=5 --batch-size=32 --lr=0.01 --arch=attention --epochs=800 --multi_chan --data=./dy016/
python run.py --out_dim=5 --proj_dim=5 --batch-size=32 --lr=0.01 --arch=attention --epochs=800 

for bs in 64 128 256 512
do
for lr in 0.001 
do
python run.py --add_prefix=0405 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 "
done
done

for bs in 64 128 256 512
do
for lr in 0.001 
do
python run.py --add_prefix=0405single --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 "
done
done


python run.py --out_dim=5 --proj_dim=5 --batch-size=512 --lr=0.01 --arch=attention --epochs=800


for bs in 64 128 256 512
do
for lr in 0.001 
do
python run.py --add_prefix=0410 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 "
done
done

python run.py --out_dim=5 --proj_dim=5 --batch-size=512 --lr=0.001 --arch=attention --epochs=800 --exp=test1 
--fp16

python run.py --out_dim=5 --proj_dim=5 --batch-size=5120 --lr=0.001 --arch=attention --epochs=800 --exp=test2 --fp16 --use_gpt


for bs in 512
do
for lr in 0.001 0.01 0.0001
do
python run.py --add_prefix=0415 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt "
done
done

for bs in 512
do
for lr in 0.001 0.01 0.0001 
do
python run.py --add_prefix=0415 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal "
done
done

for bs in 512
do
for lr in 0.001 0.01 0.0001
do
python run.py --add_prefix=0416 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --n_embd=32 "
done
done

for bs in 512
do
for lr in 0.001 0.01 0.0001 
do
python run.py --add_prefix=0416 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 "
done
done

for bs in 128
do
for lr in 0.001 0.01 0.0001
do
python run.py --add_prefix=0418 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt "
done
done

for bs in 128
do
for lr in 0.001 0.01 0.0001 
do
python run.py --add_prefix=0416 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal "
done
done

for bs in 128
do
for lr in 0.001 0.01 0.0001
do
python run.py --add_prefix=0416 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --n_embd=32 "
done
done

for bs in 128
do
for lr in 0.001 0.01 0.0001 
do
python run.py --add_prefix=0418 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 "
done
done


python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --exp=gpt_test22 --fp16 --use_gpt --n_embd=32

python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 --exp=gpt_testallaug

for bs in 128 512
do
for lr in 0.001 
do
for nembd in 32 64
do
python run.py --add_prefix=0428 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} "
done
done
done


for bs in 512
do
for lr in 0.001 
do
for nembd in 32 64
do
python run.py --add_prefix=0427 --submit --arg_str="--no_collide --out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} "
done
done
done

python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 --exp=gpt_noccollide --no_collide

python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 --exp=gpt_bs128_lr0.001_n32

python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=64 --exp=gpt_bs128_lr0.001_n64

python run.py --out_dim=5 --proj_dim=5 --batch-size=128 --lr=0.001 --arch=attention --epochs=800 --fp16 --use_gpt --is_causal --n_embd=64 --exp=test

for bs in 512 128
do
for lr in 0.001 0.005 0.0003
do
for nembd in 32 64
do
python run.py --add_prefix=0501 --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} "
done
done
done

for bs in 512 
do
for lr in 0.001 0.005 0.0003
do
for nembd in 32 
do
for dim in 2 11
do
python run.py --add_prefix=0501 --submit --arg_str="--out_dim=${dim} --proj_dim=${dim} --batch-size=${bs} --lr=${lr} --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} "
done
done
done
done

for bs in 512 
do
for lr in 0.001 0.0003
do
for nembd in 32 
do
for dim in 2 5 11
do
python run.py --add_prefix=0504 --submit --arg_str="--out_dim=${dim} --proj_dim=${dim} --batch-size=${bs} --lr=${lr} --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} --add_train "
done
done
done
done

for bs in 512 
do
for lr in 0.001 
do
for nembd in 32 
do
for dim in 5
do
python run.py --add_prefix=0505 --submit --arg_str="--out_dim=${dim} --proj_dim=${dim} --batch-size=${bs} --lr=${lr} --epochs=800 --fp16 --use_gpt --is_causal --n_embd=${nembd} --eval_knn_every_n_epochs=790 "
done
done
done
done

python \
      /gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/main.py \
      --data /gpfs/wscgpfs02/shivsr/cloh/spike_data/multi_dy016_random_neurons_04_28_2023 \
      --workers 32 \
      --epochs 100 \
      --batch-size 8 \
      --out_dim 5 \
      --proj_dim 5 \
      --optimizer adam \
      --learning-rate 0.001 \
      --checkpoint-dir /gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/saved_models/ \
      --log-dir /gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/logs/ \
      --ngpus-per-node 4 \
      --nodes 4 \
      --exp test_wsc \
      --block_size 1331 \
      --n_embd 64 \
      --multi_chan \
      --pos_enc conseq \
      --is_causal \
      --num_extra_chans 5 \
      --knn-freq 10 \
      --add_train \
      --use_chan_pos
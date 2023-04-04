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
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=${bs} --lr=${lr} --arch=attention --epochs=800 "
done
done

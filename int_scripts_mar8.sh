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


for lr in 0.001 0.0001 0.00001 0.01
do
python run.py --submit --arg_str="--out_dim=5 --proj_dim=5 --batch-size=512 --lr=${lr} --arch=attention --epochs=300"
done
from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor, gmm_monitor
import torch
import numpy as np

class Args:
    multi_chan: bool = True
    use_chan_pos: bool = False
    num_extra_chans: int = 2
    num_classes: int = 10
    block_size: int = 605
    use_gpt: bool = True
    random_state: int = 0
args = Args()

# dy016
# multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_05_16_2023'
multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_400n_goodunit_subset_06_12_2023'

train_loader = get_dataloader(multi_data_path, multi_chan=True, split='train', use_chan_pos=False, num_extra_chans=args.num_extra_chans)
test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=False, num_extra_chans=args.num_extra_chans)

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block605_bs512_extra2_lr0.0005_knn10_addtrain/checkpoint.pth'
# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0707_outdim5proj5_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0005_knn10_addtrain/checkpoint.pth'
# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0612_outdim5proj5_mc_gpt_conseq_causal_nembd64_block605_bs512_extra2_lr0.0005_knn10_addtrain/checkpoint.pth'
# multi_ckpt_path= '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0627_outdim5proj5_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0005_knn10_addtrain/checkpoint.pth'
# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0627_outdim5proj5_mc_gpt_conseq_causal_nembd64_block605_bs512_extra2_lr0.0005_knn10_addtrain/checkpoint.pth'
# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0513_outdim5proj5_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0001_knn10_addtrain_2/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, n_extra_chans=args.num_extra_chans,block_size=args.block_size, multi_chan = True, rep_after_proj=False, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=False)
save_suffix = '_eval_on_subset'

knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn:", knn_score)
gmm_score = gmm_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',args=args)
print("gmm:", gmm_score)
gmm_all = []
for i in range(10):
    # knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
    # print("knn:", knn_score)
    args.random_state = i
    gmm_score = gmm_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',args=args)
    # print("gmm:", gmm_score)
    gmm_all.append(gmm_score)

print(f"GMM: {np.mean(gmm_all)} +/- {np.std(gmm_all)}")
print(f"max GMM: {np.max(gmm_all)}")
# save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

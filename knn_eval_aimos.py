from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor
import torch

multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_04_28_2023'
multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0509_outdim512proj3_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0001_knn10_addtrain/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=True, rep_dim=512, proj_dim=3, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False)
model2 = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=False, rep_dim=512, proj_dim=3, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False)
save_suffix = '_data0428'

# multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_05_13_2023'
# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0513_outdim5proj5_mc_gpt_conseq_causal_nembd64_block1342_bs128_extra5_lr0.0001_knn10_addtrain_concatpos/checkpoint.pth'
# model = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=True, rep_dim=5, proj_dim=5, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
# model2 = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
# save_suffix = '_ep400'

train_loader = get_dataloader(multi_data_path, multi_chan=True, split='train', use_chan_pos=True)
test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=True)

class Args:
    multi_chan: bool = True
    use_chan_pos: bool = True
    num_extra_chans: int = 5
args = Args()

save_reps(model2.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=True, use_chan_pos=True, suffix=save_suffix)

save_reps(model2.cuda(), train_loader, multi_ckpt_path, split='train', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)
save_reps(model.cuda(), train_loader, multi_ckpt_path, split='train', multi_chan=True, rep_after_proj=True, use_chan_pos=True, suffix=save_suffix)

knn_score = knn_monitor(net=model2.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn if rep BEFORE proj:", knn_score)

knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn if rep AFTER proj:", knn_score)

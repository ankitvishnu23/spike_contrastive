from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor
import torch

class Args:
    multi_chan: bool = True
    use_chan_pos: bool = False
    num_extra_chans: int = 2
    num_classes: int = 10
args = Args()

# dy016
# multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_05_16_2023'
multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_400n_goodunit_subset_06_12_2023'

train_loader = get_dataloader(multi_data_path, multi_chan=True, split='train', use_chan_pos=False, num_extra_chans=args.num_extra_chans)
test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=False, num_extra_chans=args.num_extra_chans)

# multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block605_bs512_extra2_lr0.0005_knn10_addtrain/checkpoint.pth'
multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0612_outdim5proj5_mc_gpt_conseq_causal_nembd64_block605_bs512_extra2_lr0.0005_knn10_addtrain/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=605, multi_chan = True, rep_after_proj=False, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=False)
save_suffix = '_eval_on_subset'


knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn:", knn_score)

# save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor
import torch

class Args:
    multi_chan: bool = True
    use_chan_pos: bool = True
    num_extra_chans: int = 2
    num_classes: int = 400
args = Args()

# dy016
multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_05_16_2023'

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0001_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_suffix = '_eval_on_dy016_random_neurons'

test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=True, num_extra_chans=args.num_extra_chans)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0005_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

# dy016 ood
multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_OOD_UNITS_05_16_2023'

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0001_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_suffix = '_eval_on_dy016_random_OOD_units'

test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=True, num_extra_chans=args.num_extra_chans)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0005_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)


# dy09 
multi_data_path='/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy09_10_random_neurons_test+train_05_17_2023'

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0001_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_suffix = '_eval_on_dy09_10_random_neurons'

test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=True, num_extra_chans=args.num_extra_chans)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

multi_ckpt_path = '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_contrastive/saved_models/0519_outdim5proj5_mc_gpt_conseq_causal_nembd64_block610_bs512_extra2_lr0.0005_concatpos/checkpoint.pth'
model = load_ckpt(multi_ckpt_path, block_size=610, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=False, add_layernorm=False, half_embed_each=False, concat_pos=True)
save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True, suffix=save_suffix)

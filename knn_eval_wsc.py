from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor
import torch

multi_data_path='/gpfs/wscgpfs02/shivsr/cloh/spike_data/multi_dy016_random_neurons_04_28_2023'

# multi_ckpt_path = '/gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/saved_models/0510_mc_conseq_causal_n64_b1331_bs120_extra5_lr0.0005_poschan/checkpoint_epoch560.pth'
# model = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=True, use_chan_pos=True)
# model2 = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=False, use_chan_pos=True)

multi_ckpt_path = '/gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/saved_models/0510_mc_conseq_causal_n64_b1331_bs120_extra5_lr0.0005_poschan_mergelayer_layernorm/checkpoint_epoch500.pth'
model = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=True, use_chan_pos=True, use_merge_layer=True, add_layernorm=True)
model2 = load_ckpt(multi_ckpt_path, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=True, add_layernorm=True)

train_loader = get_dataloader(multi_data_path, multi_chan=True, split='train', use_chan_pos=True)
test_loader = get_dataloader(multi_data_path, multi_chan=True, split='test', use_chan_pos=True)

class Args:
    multi_chan: bool = True
    use_chan_pos: bool = True
    num_extra_chans: int = 5
args = Args()

save_reps(model.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True)
save_reps(model.cuda(), train_loader, multi_ckpt_path, split='train', multi_chan=True, rep_after_proj=False, use_chan_pos=True)


save_reps(model2.cuda(), test_loader, multi_ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True)
save_reps(model2.cuda(), train_loader, multi_ckpt_path, split='train', multi_chan=True, rep_after_proj=False, use_chan_pos=True)


knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn if rep AFTER proj:", knn_score)

knn_score = knn_monitor(net=model2.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn if rep BEFORE proj:", knn_score)
from load_models import load_ckpt, get_dataloader, save_reps
from ddp_utils import knn_monitor
import torch

data_path='/home/gridsan/evanv/charlotte/spike_data/single_dy016_random_cell_type_normalized_05_12_2023'
ckpt_path = '/home/gridsan/evanv/charlotte/spike_contrastive/runs/0513out_dim128proj_dim5batch-size512lr0.001epochs800fp16use_gptis_causaln_embd32add_traincell_type/checkpoint.pth'
model = load_ckpt(ckpt_path, rep_dim=128, proj_dim=5, multi_chan = False, rep_after_proj=True, use_chan_pos=False, use_merge_layer=False, add_layernorm=False, half_embed_each=False)
# model2 = load_ckpt(path, multi_chan = True, rep_after_proj=False, use_chan_pos=True, use_merge_layer=True, add_layernorm=True, half_embed_each=False)

train_loader = get_dataloader(data_path, multi_chan=False, split='train', use_chan_pos=False)
test_loader = get_dataloader(data_path, multi_chan=False, split='test', use_chan_pos=False)

class Args:
    multi_chan: bool = False
    use_chan_pos: bool = False
    num_extra_chans: int = 0
args = Args()


# save_reps(model2.cuda(), test_loader, ckpt_path, split='test', multi_chan=True, rep_after_proj=False, use_chan_pos=True)
save_reps(model.cuda(), test_loader, ckpt_path, split='test', multi_chan=False, rep_after_proj=True, use_chan_pos=False)

# save_reps(model2.cuda(), train_loader, multi_ckpt_path, split='train', multi_chan=True, rep_after_proj=False, use_chan_pos=True)

save_reps(model.cuda(), train_loader, ckpt_path, split='train', multi_chan=False, rep_after_proj=True, use_chan_pos=False)

# knn_score = knn_monitor(net=model2.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
# print("knn if rep BEFORE proj:", knn_score)

knn_score = knn_monitor(net=model.cuda(), memory_data_loader=train_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
print("knn if rep AFTER proj:", knn_score)

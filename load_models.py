import torch
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from models.model_GPT import GPTConfig, Multi_GPT, Single_GPT, Projector
from data_aug.wf_data_augs import Crop

class Encoder(torch.nn.Module):
    def __init__(self, multi_chan = False, single_rep_dim=5, pos_enc='conseq', rep_after_proj=False):
        super().__init__()
        if multi_chan:
            model_args = dict(bias=False, block_size=1331, n_layer=20, n_head =4, n_embd=64, dropout=0.2, out_dim=5, proj_dim=5, is_causal=True, pos = pos_enc, multi_chan=True)
        else:
            model_args = dict(bias=False, block_size=121, n_layer=20, n_head =4, n_embd=32, dropout=0.2, out_dim=single_rep_dim, proj_dim=single_rep_dim, is_causal=True, pos = pos_enc, multi_chan=False)
        gptconf = GPTConfig(**model_args)
        if multi_chan:
            self.backbone = Multi_GPT(gptconf)
        else:
            self.backbone = Single_GPT(gptconf)
            self.backbone.projector = torch.nn.Identity() # this is loaded separately later
        if rep_after_proj:
            self.projector = Projector(rep_dim=gptconf.out_dim, proj_dim=gptconf.proj_dim)
        else:
            self.projector = None

    def forward(self, x):
        r = self.backbone(x)
        if self.projector is not None:
            r = self.projector(r)
        return r   

def load_ckpt(ckpt_path, multi_chan=False, single_rep_dim=5, pos_enc='conseq', rep_after_proj=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = Encoder(multi_chan=multi_chan, single_rep_dim=single_rep_dim, pos_enc=pos_enc, rep_after_proj=rep_after_proj)
    if multi_chan:
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        m, uek = model.load_state_dict(state_dict, strict=False)
        if rep_after_proj:
            for k in uek:
                assert 'online_head' in k, "Error: key matching errors!"
        else:
            for k in uek:
                assert 'projector' in k or 'online' in k, "Error: key matching errors!"
    else:
        state_dict = {'backbone.'+k: v for k,v in ckpt['state_dict'].items() if 'projector' not in k}
        state_dict.update({k: v for k,v in ckpt['state_dict'].items() if 'projector' in k})
        m, uek = model.load_state_dict(state_dict, strict=False)
        if not rep_after_proj:
            for k in uek:
                assert 'projector' in k, "Error: key matching errors!"
        else:
            assert(len(uek)==0)
    assert(len(m)==0)
    print("missing keys", m)
    print("unexpected keys", uek)
    # assert that unexpected keys should only contain the string 'projector'
    
    return model

def get_dataloader(data_path, multi_chan=False, split='train'):
    if multi_chan:
        dataset = WFDataset_lab(data_path, split=split, multi_chan=True, transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True))
    else:
        dataset = WFDataset_lab(data_path, split=split, multi_chan=False)
        
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False,
            num_workers=16, pin_memory=True, drop_last=False)
    return loader

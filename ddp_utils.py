import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
import random
from PIL import Image, ImageOps, ImageFilter
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def consume_prefix_in_state_dict_if_present(
    state_dict, prefix) -> None:
    r"""Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

# get representations of data in torch tensor format
def get_torch_reps(net, data_loader, device, args):
    feature_bank = []
    feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, target in data_loader:
            if args.use_gpt:
                feature = net(data.to(device=device, non_blocking=True).unsqueeze(dim=-1))
            else:
                feature = net(data.to(device=device, non_blocking=True).unsqueeze(dim=1))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = torch.cat(torch.tensor(feature_labels, device=feature_bank.device), dim=0)
    
    return feature_bank, feature_labels


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# GMM fitting to data for validation
def gmm_monitor(net, memory_data_loader, test_data_loader, device='cuda', hide_progress=False,
                targets=None, args=None):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()
    classes = test_data_loader.num_classes

    # covariance_type : {'full', 'tied', 'diag', 'spherical'}
    covariance_type = 'full'
    reps_train, labels_train = get_torch_reps(net, memory_data_loader, device, args)
    reps_test, labels_test = get_torch_reps(net, test_data_loader, device, args)
    gmm = GaussianMixture(classes, 
                        random_state=0, 
                        covariance_type=covariance_type).fit(reps_train)
    gmm_cont_test_labels = gmm.predict(reps_test)
    score = adjusted_rand_score(labels_test, gmm_cont_test_labels)*100

    return score


# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, num_classes, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None, args=None):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()
    # classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            if not args.multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                if args.use_chan_pos:
                    data, chan_pos = data
                data = data.view(-1, int(args.num_extra_chans*2+1)*121)
                data = torch.unsqueeze(data, dim=-1)
            
            if args.use_chan_pos:
                feature = net(data.to(device=device, non_blocking=True), chan_pos=chan_pos.to(device=device, non_blocking=True))
            else:
                feature = net(data.to(device=device, non_blocking=True))
            
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            
            target = target.to(device=device, non_blocking=True)
            if not args.multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                if args.use_chan_pos:
                    data, chan_pos = data
                else:
                    chan_pos = None
                data = data.view(-1, int(args.num_extra_chans*2+1)*121)
                data = torch.unsqueeze(data, dim=-1)
                
            if args.use_chan_pos:
                feature = net(data.to(device=device, non_blocking=True), chan_pos=chan_pos.to(device=device, non_blocking=True))
            else:
                feature = net(data.to(device=device, non_blocking=True))

            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


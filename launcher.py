from pathlib import Path
import logging
import os
import uuid
import subprocess

import submitit
import numpy as np
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='DDP spikes training')
parser.add_argument('--data', metavar='DIR', default='/home/gridsan/cloh/spike_contrastive/dy016/',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='wfs',
                    help='dataset name', choices=['wfs', 'stl10', 'cifar10'])
parser.add_argument('--optimizer', default='adam', choices = ['adam', 'sgd'],
                    help='optimizer')
parser.add_argument('-a', '--arch', metavar='ARCH', default='custom_encoder',
                    help='default: custom_encoder)')
parser.add_argument('-ns', '--noise_scale', default=1.0,
                    help='how much to scale the noise augmentation (default: 1)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=5, type=int,
                    help='feature dimension (default: 2)')
parser.add_argument('--proj_dim', default=5, type=int,
                    help='projection dimension (default: 5)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fc_depth', default=2, type=int)
parser.add_argument('--submit', action='store_true')
parser.add_argument('--arg_str', default='--', type=str)
parser.add_argument('--add_prefix', default='', type=str)
parser.add_argument('--no_proj', default='True', action='store_true')
parser.add_argument('--expand_dim', default=16, type=int)
parser.add_argument('--multi_chan', default=False, action='store_true')
parser.add_argument('--n_channels', default=11, type=int)
parser.add_argument('--ddp', default=True, action='store_true')
parser.add_argument('--eval_knn_every_n_epochs', default=1000, type=int)

parser.add_argument('--use_gpt', action='store_true') # default = False

parser.add_argument('--n_layer', default=20, type=int)
parser.add_argument('--n_head', default=4, type=int)
parser.add_argument('--n_embd', default=64, type=int)
parser.add_argument('--is_causal', action='store_true') # default = False
# parser.add_argument('--block_size', default=2678, type=int) # this is the max sequence length
parser.add_argument('--block_size', default=1331, type=int) # this is the max sequence length

parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--bias', action='store_true') # default = False
parser.add_argument('--vocab_size', default=50304, type=int) # default to GPT-2 vocab size

parser.add_argument('--online_head', action='store_true', default=True) # default = True
    
# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str,
                    help="Partition where to submit")

parser.add_argument('--exp', default='test', type=str)
parser.add_argument('--checkpoint-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path,
                    metavar='LOGDIR', help='path to tensorboard log directory')

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import run
        self._setup_gpu_args()
        run.main_worker(self.args.gpu, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = args.job_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp
    args.job_dir = args.checkpoint_dir

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    get_init_file(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus_per_node
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    kwargs = {'slurm_gres': f'gpu:{num_gpus_per_node}',}

    executor.update_parameters(
        mem_gb=30 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=24,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 6
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.exp)

    args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()
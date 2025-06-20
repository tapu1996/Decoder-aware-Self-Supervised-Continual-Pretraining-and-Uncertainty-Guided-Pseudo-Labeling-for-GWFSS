# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch
import torch.distributed as dist


##################### DDP functions #####################
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
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

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x.contiguous())
    return torch.cat(x_list, dim=0)

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

##################### DDP setup #####################
############################################### TO BE USED WITH TORCH RUN ###############################################
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ["RANK"] = str(args.rank)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


#########################################################################################################################


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _init_distributed_mode_sagemaker(**kwargs):

    # SageMaker data parallel: Initialize the process group
    dist.init_process_group(backend="smddp")

    rank = dist.get_rank()  # int(os.environ['LOCAL_RANK'])
    if rank == 0:
        print(
            "WARNING: This ddp trainer is designed for AWS sagemaker platform",
            "Initializing distributed mode",
        )
        # In an other context, we could need to use these commands
        # self.dist_backend = 'nccl'
        # torch.distributed.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
        #                                      world_size=self.world_size, rank=self.rank)

    print("RANK : ", rank)
    # self.world_size is already defined by AWS instance type
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    print("LOCAL_RANK : ", local_rank)
    world_size = dist.get_world_size()
    print(
        "| distributed init (rank {}): local_rank {}".format(rank, local_rank),
        flush=True,
    )

    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def _init_distributed_mode_computecan(**kwargs):

    # Example interactive allocation to use DDP on CC:
    # salloc --nodes 1 --time=03:00:00 --account=def-senger --cpus-per-task=11 --mem=64G --gres=gpu:2 --tasks-per-node=2
    # To tell the system you want to use NCCL back end for inter-GPUs communication
    os.system("export NCCL_BLOCKING_WAIT=1")
    # init_method = "tcp://$MASTER_ADDR:3456"
    print("Getting master address")
    mastr_add = str(os.system("hostname"))  # str(os.environ.get("MASTER_ADDR"))
    print("Setting init method")
    init_method = f"tcp://{mastr_add}:3456"
    # world_size to retrieve from $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))
    print("Setting world size")
    world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE")) * int(
        os.environ.get("SLURM_JOB_NUM_NODES")
    )
    # export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the
    # MASTER_ADDR environment variable.
    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""
    # echo "r$SLURM_NODEID master: $MASTER_ADDR"
    # echo "r$SLURM_NODEID Launching python script"
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

    current_device = local_rank

    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print("From Rank: {}, ==> Initializing Process Group...".format(rank))
    # init the process group
    # export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish
    # to use the NCCL backend for inter-GPU communication.
    dist.init_process_group(
        backend="nccl", init_method=init_method, world_size=world_size, rank=rank
    )
    print("process group ready!")

    print("From Rank: {}, ==> Making model..".format(rank))

    return local_rank, rank, world_size


dist_init_functions = {
    "sage-maker-env": _init_distributed_mode_sagemaker,
    "cc-env": _init_distributed_mode_computecan,
}
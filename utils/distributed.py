import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Union, Any, Tuple

__all__ = [
    "init_distributed",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "synchronize",
    "all_reduce_tensor",
    "all_gather_tensor",
    "reduce_loss",
    "prepare_distributed_dataloader",
    "cleanup_distributed"
]


def init_distributed(
    backend: str = "nccl",
    port: str = "29500",
    timeout: int = 3600
) -> Tuple[int, int, int]:
    if "SLURM_PROCID" in os.environ:
        # SLURM cluster environment
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(",")[0]
        os.environ["MASTER_PORT"] = port
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Torch distributed launch
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single process, no distributed
        return 0, 0, 1

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=dist.default_pg_timeout if timeout is None else dist.Timeout(seconds=timeout)
    )

    # Set default device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    synchronize()
    return rank, local_rank, world_size


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_distributed():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    if not is_distributed():
        return
    dist.barrier()


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    average: bool = True
) -> torch.Tensor:
    if not is_distributed():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    if average:
        tensor.div_(get_world_size())
    return tensor


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous())
    return torch.cat(tensor_list, dim=0)


def reduce_loss(loss: torch.Tensor, average: bool = True) -> torch.Tensor:
    return all_reduce_tensor(loss, average=average)


def prepare_distributed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )
    return dataloader, sampler


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# Runtime verification
if __name__ == "__main__":
    rank, local_rank, world_size = init_distributed()
    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    print(f"Is main process: {is_main_process()}")
    print(f"Is distributed: {is_distributed()}")

    # Test all reduce
    test_tensor = torch.tensor([1.0], device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    reduced_tensor = all_reduce_tensor(test_tensor, average=True)
    print(f"Original tensor: {test_tensor.item()}, Reduced tensor: {reduced_tensor.item()}")

    synchronize()
    cleanup_distributed()
    print("Distributed utils test completed.")

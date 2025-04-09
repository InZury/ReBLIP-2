# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import datetime
import functools
import os
import torch
import torch.distributed as dist
import timm.models._hub as hub

from lavis.utils.logger import logger


def setup_for_distributed(is_main):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def dist_print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = dist_print


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


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logger.warning("Not using distributed mode!")
        args.distributed = False

        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    logger.info(f"distributed init (rank {args.rank}, world {args.world_size}): {args.dist_url}")

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(days=365)
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_dist_info():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()

        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def download_cached_file(url, check_hash=True, progress=False):
    """
        Download a file from a URL and cache it locally.

        If the file already exists, it is not downloaded again.

        If distributed, only the main process downloads the file,
        and the other processes wait for the file to be downloaded.
    """
    def get_cached_file_path():
        parts = torch.hub.urlparse(url)
        file_name = os.path.basename(parts.path)
        cached_file = os.path.join(hub.get_cache_dir(), file_name)

        return cached_file

    if is_main_process():
        hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()

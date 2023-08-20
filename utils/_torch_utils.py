import gc

from utils import cuda_is_available, package_exists


def torch_optimizer(func):
    import torch

    def wrapped(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast("cuda"):
                return func(*args, **kwargs)

    return wrapped


def cuda_garbage_collection():
    gc.collect()
    if cuda_is_available() and package_exists("torch"):
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

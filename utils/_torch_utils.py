import gc
from typing import Optional

from utils import cuda_is_available, package_exists


def torch_optimizer(func):
    import torch

    def wrapped(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast("cuda"):
                return func(*args, **kwargs)

    return wrapped


def cuda_garbage_collection(func:Optional[callable]=None):
    def _cuda_garbage_collection(*args, **kwargs):
        if cuda_is_available() and package_exists("torch"):
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        finally:
            _cuda_garbage_collection()
        return ret

    if func is not None:
        return wrapper
    else:
        _cuda_garbage_collection()

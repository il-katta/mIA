import gc


def torch_optimizer(func):
    import torch

    def wrapped(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast("cuda"):
                return func(*args, **kwargs)

    return wrapped


def cuda_garbage_collection():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

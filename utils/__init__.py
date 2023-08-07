import importlib.util
import gc


def cuda_garbage_collection():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def package_exists(package_name: str) -> bool:
    """
    Check if a package exists in the current environment.
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

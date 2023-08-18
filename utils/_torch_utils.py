
def torch_optimizer(func):
    import torch
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast("cuda"):
                return func(*args, **kwargs)

    return wrapped
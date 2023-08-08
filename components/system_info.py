import gradio as gr

from utils import package_exists


def is_available():
    return package_exists("torch")


def gui():
    import torch

    def get_gpu_ram_usage(device_id=0):
        allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
        return f"#{device_id} | {allocated} / {total} GB"

    def get_gpu_summary(device_id=0):
        return torch.cuda.memory_summary(device_id, abbreviated=False)

    def get_gpu_name(device_id=0):
        return torch.cuda.get_device_name(device_id)

    with gr.Row():
        for gpu_id in range(torch.cuda.device_count()):
            with gr.Tab(label=get_gpu_name(gpu_id)):
                # GPU ram usage
                gr.Markdown(value=lambda: get_gpu_ram_usage(gpu_id), label="GPU RAM usage", every=2)
                # GPU summary
                #gr.Markdown(value=lambda: get_gpu_summary(gpu_id), label="GPU summary", every=2)

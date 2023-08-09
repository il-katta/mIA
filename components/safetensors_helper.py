from utils import package_exists
import gradio as gr
import json

from utils.system_stats import SystemStats


def is_available():
    return package_exists("torch")


def gui(sysstats: SystemStats):
    from utils.safetensors_utils import SafetensorsUtils

    def read_metadata(filepath) -> str:
        return json.dumps(SafetensorsUtils.read_metadata(filepath))

    with gr.Row():
        gr.File()
        filepath_text = gr.Textbox(placeholder="Safetensor file path", label="Filepath", interactive=True)
    with gr.Row():
        submit_button = gr.Button("Read metadata")

    with gr.Row():
        metadata_text = gr.Json(label="Metadata", interactive=False)

    submit_button.click(
        read_metadata,
        inputs=[filepath_text],
        outputs=[metadata_text],
        api_name="safe_tensors_read_metadata"
    )
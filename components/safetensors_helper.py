from utils import package_exists
import gradio as gr
import json


def is_available():
    return package_exists("torch")


def gui():
    from utils.safetensors_utils import SafetensorsUtils

    def read_metadata(filepath) -> str:
        return json.dumps(SafetensorsUtils.read_metadata(filepath))

    with gr.Row():
        gr.File()
        filepath_text = gr.Textbox(placeholder="Safetensor file path", label="Filepath", interactive=True)
    with gr.Row():
        submit_button = gr.Button("Read metadata")

    with gr.Row():
        metadata_text = gr.Json(label="Metadata", lines=99, interactive=False)

    submit_button.click(read_metadata, inputs=[filepath_text], outputs=[metadata_text])
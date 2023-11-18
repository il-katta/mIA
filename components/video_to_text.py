import gradio as gr

from utils import package_exists
from utils.system_stats import SystemStats
from utils.video2text import Video2Text


def is_available():
    return package_exists("yt_dlp") and package_exists("whisper")


def gui(sysstats: SystemStats):
    v2t = Video2Text.get_instance()
    sysstats.register_disposable_model(v2t)

    with gr.Row():
        url = gr.Textbox(label="URL", lines=1, placeholder="URL")
        model_name_dropdown = gr.Dropdown(
            Video2Text.available_models(), value=Video2Text.available_models()[0],
            label="Model Name",
        )
    with gr.Row():
        execute_button = gr.Button("Execute")
    with gr.Row():
        video_transcript = gr.Textbox(label="Video transcript", lines=100, placeholder="Video transcript")

    def process_video(video_url: str, model_name: str) -> str:
        video_text = v2t.video2text(video_url, model_name)
        return video_text

    execute_button.click(
        process_video,
        inputs=[url, model_name_dropdown],
        outputs=[video_transcript],
    )

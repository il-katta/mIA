# noinspection PyUnresolvedReferences
import utils._force_load_env
from utils import is_debug_mode_enabled
from utils.system_stats import SystemStats
import logging
import os

import gradio as gr

from components import (
    settings, chat, music_images_generator, remove_backgroud, image_upscale, invisible_watermark,
    generate_music, generate_sounds, safetensors_helper, generate_image, edit_image, system_info
)
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)

logging.getLogger("gradio").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

os.makedirs(config.DATA_DIR, exist_ok=True)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

with gr.Blocks() as demo:
    conf = config.load_config()
    sysstats: SystemStats = SystemStats(store_history=True)

    if chat.is_available():
        with gr.Tab("Chat"):
            chat.gui(conf=conf, sysstats=sysstats)

    if generate_image.is_available():
        with gr.Tab("Image Generator"):
            generate_image.gui(sysstats=sysstats)

    if music_images_generator.is_available():
        with gr.Tab("Music Images Generator"):
            music_images_generator.gui(sysstats=sysstats)

    if edit_image.is_available():
        with gr.Tab("Image Editor"):
            edit_image.gui(sysstats=sysstats)

    if remove_backgroud.is_available():
        with gr.Tab("Image background remover"):
            remove_backgroud.gui(sysstats=sysstats)

    if image_upscale.is_available():
        with gr.Tab("Image upscaler"):
            image_upscale.gui(sysstats=sysstats)

    if invisible_watermark.is_available():
        with gr.Tab("Invisible Watermark"):
            invisible_watermark.gui(sysstats=sysstats)

    if generate_music.is_available():
        with gr.Tab("Music Generator"):
            generate_music.gui(sysstats=sysstats)

    if generate_sounds.is_available():
        with gr.Tab("Sound Generator"):
            generate_sounds.gui(sysstats=sysstats)

    if safetensors_helper.is_available():
        with gr.Tab("Read safetensors metadata"):
            safetensors_helper.gui(sysstats=sysstats)

    with gr.Tab("Settings"):
        settings.gui(conf=conf, sysstats=sysstats)

    if system_info.is_available():
        with gr.Accordion("System Info", open=False):
            system_info.gui(sysstats=sysstats)

gr.close_all(True)
demo.queue(concurrency_count=10)

if __name__ == "__main__":
    import signal
    import sys


    def on_signal(sig, frame):
        print("Bye")
        if demo.enable_queue:
            try:
                demo._queue.close()
                demo._queue.push(None)
            except:
                pass
        demo.close()
        sys.exit(0)


    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        if is_debug_mode_enabled():
            demo.launch(server_name="0.0.0.0", debug=True, show_error=True, app_kwargs={"dev": "true"}, server_port=1988)
        else:
            demo.launch(server_name="0.0.0.0", show_error=True, server_port=1988)
    except KeyboardInterrupt:
        on_signal(None, None)

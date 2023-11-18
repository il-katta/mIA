# noinspection PyUnresolvedReferences
import utils._force_load_env
from utils import is_debug_mode_enabled
from utils.system_stats import SystemStats
import logging
import os

import gradio as gr

from components import (
    settings, chat, music_images_generator, remove_backgroud, image_upscale, invisible_watermark,
    generate_music, generate_sounds, safetensors_helper, generate_image, edit_image, random_audio, image_tagger,
    video_to_text,
    system_info
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
        logging.info("Chat is available")
        with gr.Tab("Chat"):
            chat.gui(conf=conf, sysstats=sysstats)
    else:
        logging.warning("Chat is not available")

    if generate_image.is_available():
        logging.info("Image Generator is available")
        with gr.Tab("Image Generator"):
            generate_image.gui(sysstats=sysstats)
    else:
        logging.warning("Image Generator is not available")

    if music_images_generator.is_available():
        logging.info("Music Images Generator is available")
        with gr.Tab("Music Images Generator"):
            music_images_generator.gui(sysstats=sysstats)
    else:
        logging.warning("Music Images Generator is not available")

    if edit_image.is_available():
        logging.info("Image Editor is available")
        with gr.Tab("Image Editor"):
            edit_image.gui(sysstats=sysstats)
    else:
        logging.warning("Image Editor is not available")

    if remove_backgroud.is_available():
        logging.info("Image background remover is available")
        with gr.Tab("Image background remover"):
            remove_backgroud.gui(sysstats=sysstats)
    else:
        logging.warning("Image background remover is not available")

    if image_upscale.is_available():
        logging.info("Image upscaler is available")
        with gr.Tab("Image upscaler"):
            image_upscale.gui(sysstats=sysstats)
    else:
        logging.warning("Image upscaler is not available")

    if invisible_watermark.is_available():
        logging.info("Invisible Watermark is available")
        with gr.Tab("Invisible Watermark"):
            invisible_watermark.gui(sysstats=sysstats)
    else:
        logging.warning("Invisible Watermark is not available")

    if generate_music.is_available():
        logging.info("Music Generator is available")
        with gr.Tab("Music Generator"):
            generate_music.gui(sysstats=sysstats)
    else:
        logging.warning("Music Generator is not available")

    if generate_sounds.is_available():
        logging.info("Sound Generator is available")
        with gr.Tab("Sound Generator"):
            generate_sounds.gui(sysstats=sysstats)
    else:
        logging.warning("Sound Generator is not available")

    if safetensors_helper.is_available():
        logging.info("Safetensors helper is available")
        with gr.Tab("Read safetensors metadata"):
            safetensors_helper.gui(sysstats=sysstats)
    else:
        logging.warning("Safetensors helper is not available")

    if random_audio.is_available():
        logging.info("Random audio Generator is available")
        with gr.Tab("Random audio Generator"):
            random_audio.gui(sysstats=sysstats)
    else:
        logging.warning("Random audio Generator is not available")

    if image_tagger.is_available():
        logging.info("Image Tagger is available")
        with gr.Tab("Image Tagger"):
            image_tagger.gui(sysstats=sysstats)
    else:
        logging.warning("Image Tagger is not available")

    if video_to_text.is_available():
        logging.info("Video to Text is available")
        with gr.Tab("Video to Text"):
            video_to_text.gui(sysstats=sysstats)
    else:
        logging.warning("Video to Text is not available")

    with gr.Tab("Settings"):
        settings.gui(conf=conf, sysstats=sysstats)

    if system_info.is_available():
        logging.info("System Info is available")
        with gr.Accordion("System Info", open=False):
            system_info.gui(sysstats=sysstats)
    else:
        logging.warning("System Info is not available")

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

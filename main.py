from dotenv import load_dotenv

load_dotenv()

import logging
import os

import gradio as gr

from bot import MiaBot
from components import (
    settings, chat, music_images_generator, remove_backgroud, image_upscale, invisible_watermark,
    generate_music, generate_sounds, safetensors_helper, system_info
)
from utils.tts import TextToSpeech
import config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
os.makedirs(config.DATA_DIR, exist_ok=True)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

with gr.Blocks() as demo:
    conf = config.load_config()
    bot = MiaBot(conf)
    tts = TextToSpeech(config.ELEVENLABS_DEFAULT_APIKEY)

    hello_text = gr.Label("Hello there! I'm mIA, your personal assistant. How can I help you?")

    if system_info.is_available():
        system_info.gui()

    if chat.is_available():
        with gr.Tab("Chat"):
            chat.gui(bot=bot, conf=conf)

    if music_images_generator.is_available():
        with gr.Tab("Music Images Generator"):
            music_images_generator.gui(bot=bot, conf=conf)

    if remove_backgroud.is_available():
        with gr.Tab("Image background remover"):
            remove_backgroud.gui()

    if image_upscale.is_available():
        with gr.Tab("Image upscaler"):
            image_upscale.gui()

    if invisible_watermark.is_available():
        with gr.Tab("Invisible Watermark"):
            invisible_watermark.gui()

    if generate_music.is_available():
        with gr.Tab("Music Generator"):
            generate_music.gui()

    if generate_sounds.is_available():
        with gr.Tab("Sound Generator"):
            generate_sounds.gui()

    if safetensors_helper.is_available():
        with gr.Tab("Safetensors Helper"):
            safetensors_helper.gui()

    with gr.Tab("Settings"):
        settings.gui(tts=tts, conf=conf)

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, show_error=True, app_kwargs={"dev": "true"}, server_port=1988)

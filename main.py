import logging
import os

import gradio as gr
from dotenv import load_dotenv

from bot import MiaBot
from components import settings, chat
from ttl import TextToVoice
import config

load_dotenv()

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
    ttl = TextToVoice(config.ELEVENLABS_DEFAULT_APIKEY)

    with gr.Tab("Chat"):
        chat.gui(bot=bot, conf=conf)

    with gr.Tab("Settings"):
        settings.gui(ttl=ttl, conf=conf)



demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, show_error=True, app_kwargs={"dev": "true"})

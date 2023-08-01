import json
import os
from abc import ABC
from pathlib import Path
from typing import Optional, Iterable, Iterator
import openai
import gradio as gr

__all__ = [
    "DATA_DIR",
    "OPENAI_API_KEY",
    "ELEVENLABS_DEFAULT_VOICE",
    "BARK_DEFAULT_VOICE",
    "GENERATOR_ELEVENLABS",
    "GENERATOR_BARK",
    "GENERATOR_DISABLED",
    "Config",
    "save_config",
    "load_config",
]

GENERATOR_ELEVENLABS = "ElevenLabs"
GENERATOR_BARK = "Bark"
GENERATOR_DISABLED = "[Disabled]"
OPENAI_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.7
DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data")))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_DEFAULT_APIKEY = os.environ.get("ELEVENLABS_API_KEY", None)

ELEVENLABS_DEFAULT_VOICE = os.environ.get("ELEVENLABS_DEFAULT_VOICE", '21m00Tcm4TlvDq8ikWAM')
BARK_DEFAULT_VOICE = os.environ.get("BARK_DEFAULT_VOICE", "v2/it_speaker_9")



class Config(Iterable[gr.State], ABC):
    openai_model_state: gr.State
    ttl_generator_state: gr.State
    elevenlabs_voice_id_state: gr.State
    bark_voice_id_state: gr.State
    openai_temperature_state: gr.State

    def __init__(
            self,
            openai_model: Optional[str] = OPENAI_MODEL,
            openai_temperature: Optional[float] = OPENAI_TEMPERATURE,
            ttl_generator: Optional[str] = GENERATOR_DISABLED,
            elevenlabs_voice_id: Optional[str] = ELEVENLABS_DEFAULT_VOICE,
            bark_voice_id: Optional[str] = BARK_DEFAULT_VOICE
    ):
        self.openai_model_state = gr.State(openai_model)
        self.openai_temperature_state = gr.State(openai_temperature)
        self.ttl_generator_state = gr.State(ttl_generator)
        self.elevenlabs_voice_id_state = gr.State(elevenlabs_voice_id)
        self.bark_voice_id_state = gr.State(bark_voice_id)
        openai.api_key = OPENAI_API_KEY

    def __list__(self) -> Iterable[gr.State]:
        return [
            self.openai_model_state,
            self.openai_temperature_state,
            self.ttl_generator_state,
            self.elevenlabs_voice_id_state,
            self.bark_voice_id_state
        ]

    def __iter__(self) -> Iterator[gr.State]:
        return iter(self.__list__())


def save_config(conf: Config):
    with open(str(DATA_DIR / "config.json"), 'w') as fd:
        json.dump({
            "openai_model": conf.openai_model_state.value,
            "openai_temperature": conf.openai_temperature_state.value,
            "ttl_generator": conf.ttl_generator_state.value,
            "elevenlabs_voice_id": conf.elevenlabs_voice_id_state.value,
            "bark_voice_id": conf.bark_voice_id_state.value,
        }, fd)


def load_config() -> Config:
    config_file = str(DATA_DIR / "config.json")
    default_config = Config()
    if not os.path.exists(config_file):
        return default_config
    else:
        with open(config_file, 'r') as fd:
            data = json.load(fd)
            return Config(
                openai_model=data.get("openai_model", default_config.openai_model_state.value),
                openai_temperature=float(data.get("openai_temperature", default_config.openai_temperature_state.value)),
                ttl_generator=data.get("ttl_generator", default_config.ttl_generator_state.value),
                elevenlabs_voice_id=data.get("elevenlabs_voice_id", default_config.elevenlabs_voice_id_state.value),
                bark_voice_id=data.get("bark_voice_id", default_config.bark_voice_id_state.value),
            )

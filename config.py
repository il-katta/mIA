import json
import logging
import os
from abc import ABC
from pathlib import Path
from typing import Optional, Iterable, Iterator, Any
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
    "load_config",
]

GENERATOR_ELEVENLABS = "ElevenLabs"
GENERATOR_BARK = "Bark"
BARK_DEVICE = "cuda"
GENERATOR_DISABLED = "[Disabled]"
OPENAI_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.7
DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data")))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_DEFAULT_APIKEY = os.environ.get("ELEVENLABS_API_KEY", None)

ELEVENLABS_DEFAULT_VOICE = os.environ.get("ELEVENLABS_DEFAULT_VOICE", '21m00Tcm4TlvDq8ikWAM')
BARK_DEFAULT_VOICE = os.environ.get("BARK_DEFAULT_VOICE", "v2/it_speaker_9")

BARK_DEVICES = [
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "ort",
    "xla",
    "lazy",
    "vulkan",
    "mps",
    "meta",
    "hpu",
    "mtia"
]


class State(gr.State, ABC):
    def __init__(self, name, default_value=None):
        self.stateful = True
        self._name = name
        self._default_value = default_value
        self._config_file = str(DATA_DIR / "config.json")
        super().__init__(self._get_value)

    @property
    def value(self):
        return self._get_value()

    def _get_value(self):
        if os.path.exists(self._config_file):
            with open(self._config_file, "r") as f:
                config = json.load(f)
                if self._name in config:
                    val = config[self._name]
                    logging.info(f"{self._name}: {val}")
                    return val
        return self._default_value

    @value.setter
    def value(self, value):
        if os.path.exists(self._config_file):
            with open(self._config_file, "r") as f:
                config = json.load(f)
        else:
            config = {}
        config[self._name] = value
        with open(self._config_file, "w") as f:
            json.dump(config, f)

    def preprocess(self, x: Any) -> Any:
        return x

    def postprocess(self, y):
        return y

    def get_block_name(self) -> str:
        return 'state'


class Config(gr.State, Iterable[gr.State], ABC):
    openai_model_state: State
    tts_generator_state: State
    elevenlabs_voice_id_state: State
    bark_voice_id_state: State
    bark_device_state: State
    openai_temperature_state: State

    def __init__(self):
        self.openai_model_state = State("openai_model", OPENAI_MODEL)
        self.openai_temperature_state = State("openai_temperature", OPENAI_TEMPERATURE)
        self.tts_generator_state = State("tts_generator", GENERATOR_DISABLED)
        self.elevenlabs_voice_id_state = State("elevenlabs_voice_id", ELEVENLABS_DEFAULT_VOICE)
        self.bark_voice_id_state = State("bark_voice_id", BARK_DEFAULT_VOICE)
        self.bark_device_state = State("bark_device", BARK_DEVICE)
        openai.api_key = OPENAI_API_KEY
        gr.components.IOComponent.__init__(self, value=self.__dict__())

    def update(
            self,
            openai_model: Optional[str] = OPENAI_MODEL,
            openai_temperature: Optional[float] = OPENAI_TEMPERATURE,
            tts_generator: Optional[str] = GENERATOR_DISABLED,
            elevenlabs_voice_id: Optional[str] = ELEVENLABS_DEFAULT_VOICE,
            bark_voice_id: Optional[str] = BARK_DEFAULT_VOICE,
            bark_device: Optional[str] = BARK_DEVICE
    ):
        self.openai_model_state.value = openai_model
        self.tts_generator_state.value = tts_generator
        self.openai_temperature_state.value = openai_temperature
        self.tts_generator_state.value = tts_generator
        self.elevenlabs_voice_id_state.value = elevenlabs_voice_id
        self.bark_voice_id_state.value = bark_voice_id
        self.bark_device_state.value = bark_device

    def __list__(self) -> Iterable[gr.State]:
        return [
            self.openai_model_state,
            self.openai_temperature_state,
            self.tts_generator_state,
            self.elevenlabs_voice_id_state,
            self.bark_voice_id_state,
            self.bark_device_state
        ]

    def __iter__(self) -> Iterator[gr.State]:
        return iter(self.__list__())

    def __dict__(self):
        return {
            "openai_model": self.openai_model_state.value,
            "openai_temperature": self.openai_temperature_state.value,
            "tts_generator": self.tts_generator_state.value,
            "elevenlabs_voice_id": self.elevenlabs_voice_id_state.value,
            "bark_voice_id": self.bark_voice_id_state.value,
            "bark_device": self.bark_device_state.value,
        }

    def get_block_name(self) -> str:
        return 'state'


def load_config() -> Config:
    return Config()

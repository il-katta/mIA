import json
import logging
import os.path
import tempfile
from typing import Optional, Dict, List

import elevenlabs

import utils

try:
    from bark import SAMPLE_RATE
    from bark.generation import CUR_PATH

    bark_available = (
            utils.package_exists("transformers") and
            utils.package_exists("bitsandbytes") and
            utils.package_exists("torch") and
            utils.package_exists("accelerate") and
            utils.package_exists("scipy")
    )
except ImportError:
    SAMPLE_RATE = 24_000
    bark_available = False
    CUR_PATH = None

import config


class TextToSpeech:

    def __init__(self, elevenlabs_apikey: Optional[str] = None):
        self._logger = logging.getLogger("TextToSpeech")
        self._logger.debug("TextToSpeech initialized")
        self._elevenlabs_apikey = elevenlabs_apikey
        if elevenlabs_apikey:
            elevenlabs.set_api_key(self._elevenlabs_apikey)

    @staticmethod
    def elevenlabs_generate(text: str, voice_id: str) -> str:
        audiofile = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        for chunk in elevenlabs.generate(
                text,
                model="eleven_multilingual_v1",
                voice=voice_id,
                stream=True
        ):
            audiofile.write(chunk)

        return audiofile.name

    @staticmethod
    def elevenlabs_voices(force_fresh=False) -> Dict[str, str]:
        cache_file = config.DATA_DIR / "voices.json"
        if os.path.exists(cache_file) and not force_fresh:
            values = json.load(open(cache_file, "r"))
        else:
            voices = elevenlabs.voices()
            values = {
                v.voice_id: f"{v.name} ({v.voice_id})"
                for v in voices.items
            }

            json.dump(values, open(cache_file, "w"))

        return values

    @staticmethod
    def bark_available():
        return bark_available

    @staticmethod
    def static_bark_init(device="cpu"):
        from bark import preload_models
        preload_models(
            text_use_gpu=False if device == "cpu" else True,
            coarse_use_gpu=False if device == "cpu" else True,
            fine_use_gpu=False if device == "cpu" else True,
            codec_use_gpu=False if device == "cpu" else True,
        )

    @classmethod
    def static_bark_generate(cls, text: str, voice_id: str, text_temp: float = 0.7, waveform_temp: float = 0.7,
                             device="cpu") -> str:
        from bark import generate_audio
        cls.static_bark_init(device=device)
        audio_array = generate_audio(text, history_prompt=voice_id, text_temp=text_temp, waveform_temp=waveform_temp)
        return cls.bark_audio_array_to_file(audio_array)

    @staticmethod
    def bark_audio_array_to_file(audio_array, sample_rate=SAMPLE_RATE) -> str:
        from scipy.io.wavfile import write as write_wav
        audiofile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        write_wav(audiofile, sample_rate, audio_array)
        return audiofile.name

    _bark_processor = None
    _bark_model = None

    def bark_init(self, device="cpu", load_in_4bit=False):
        from transformers import AutoProcessor, BarkModel, BarkProcessor
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch
                from bitsandbytes.cuda_setup.main import evaluate_cuda_setup
                evaluate_cuda_setup()
                import accelerate
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            except Exception as ex:
                self._logger.error("Could not import/setup bitsandbytes")
                self._logger.exception(ex)
                load_in_4bit = False
                bnb_config = None
        else:
            bnb_config = None

        if self._bark_processor is None:
            self._bark_processor = AutoProcessor.from_pretrained("suno/bark")
        if self._bark_model is None:
            if load_in_4bit:
                self._bark_model = BarkModel.from_pretrained("suno/bark", quantization_config=bnb_config,
                                                             device_map='auto')
            else:
                self._bark_model = BarkModel.from_pretrained("suno/bark").to(device)
        # self._bark_model.generation_config.coarse_acoustics_config.temperature
        # self._bark_model.generation_config.fine_acoustics_config.temperature
        # self._bark_model.generation_config.semantic_config.temperature

    def bark_generate(self, text: str, voice_id: str, text_temp: float = 0.7, waveform_temp: float = 0.7,
                      device="cpu") -> str:
        self.bark_init(device=device)
        inputs = self._bark_processor(text, voice_preset=voice_id).to(device)
        audio_array = self._bark_model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return self.bark_audio_array_to_file(audio_array, self._bark_model.generation_config.sample_rate)

    @staticmethod
    def bark_voices() -> List[str]:
        if not bark_available:
            return []
        _files = set()

        def _get_files(path=""):
            for (_, dirs, filenames) in os.walk(os.path.join(CUR_PATH, "assets", "prompts", path)):
                for f in filenames:
                    _files.add(os.path.join(path, f).rstrip(".npz"))
                for d in dirs:
                    _get_files(os.path.join(path, d))

        _get_files()
        files = list(_files)
        files.sort()
        return files

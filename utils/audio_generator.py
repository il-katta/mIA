import logging
from typing import Optional, Tuple, Generator

import numpy as np
from audiodiffusion import AudioDiffusion
import torch

from diffusers import AudioDiffusionPipeline

import config

from utils._interfaces import DisposableModel
from utils._torch_utils import torch_optimizer, cuda_garbage_collection

MODELS = [
    "teticio/audio-diffusion-256",
    "teticio/audio-diffusion-breaks-256",
    "teticio/audio-diffusion-instrumental-hiphop-256",
    "teticio/audio-diffusion-ddim-256",
    "teticio/latent-audio-diffusion-256",
    "teticio/latent-audio-diffusion-ddim-256"
]


class AudioGenerator(DisposableModel, AudioDiffusion):
    pipe: Optional[AudioDiffusionPipeline] = None
    model_name = None

    def __init__(self, model_name=MODELS[0]):
        self.model_name = model_name
        self.progress_bar = lambda _: _

    def load_model(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.model_name

        if model_name != self.model_name or self.pipe is None:
            self.unload_model()
            logging.info(f"Loading model {model_name}")
            self.model_name = model_name
            self.pipe = AudioDiffusionPipeline.from_pretrained(
                self.model_name,
                cache_dir=config.DATA_DIR / "huggingface"
            ).to("cuda")

    def unload_model(self):
        if self.pipe:
            logging.info(f"Unloading model {self.model_name}")
            del self.pipe
            cuda_garbage_collection()
            self.pipe = None

    @torch_optimizer
    def generate_audio(
            self,
            model_name: Optional[str] = None,
            loops: int = 1,
            steps: int = 1000,
            seed: Optional[int] = None,
            start_step: int = 0,
    ) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
        self.load_model(model_name)
        generator = torch.Generator(device="cuda")
        if seed is None or seed < 0:
            seed = generator.seed()
        generator.manual_seed(seed)
        logging.info(f"Generating audio with model {self.model_name}, seed: {seed}, steps: {steps}, loops: {loops}, start_step: {start_step}")
        images, (sample_rate, audios) = self.pipe(
            batch_size=1,
            steps=steps,
            generator=generator,
            start_step=start_step,
            return_dict=False,
        )
        audio = audios[0]
        if loops > 0:
            loop: Optional[np.ndarray] = self.loop_it(audio, sample_rate, loops=loops)  # can be None
        else:
            loop = None
        logging.info(f"Generated audio with sample rate {sample_rate}")
        if loop is not None:
            logging.info(f"Generated audio loop with sample rate {sample_rate}")
        return (sample_rate, loop), (sample_rate, audio)

    def extend_audio(
            self,
            sample_rate_audio: Tuple[int, np.ndarray],
            model_name: Optional[str] = None,
            overlap_secs: int = 2,
            loops: int = 12,
            start_step: int = 500
    ) -> Tuple[int, np.ndarray]:

        g = self._extend_audio(
                sample_rate_audio,
                model_name=model_name,
                overlap_secs=overlap_secs,
                loops=loops,
                start_step=start_step
        )

        while True:
            try:
                next(g)
            except StopIteration as e:
                return e.value

    def _extend_audio(
            self,
            sample_rate_audio: Tuple[int, np.ndarray],
            model_name: Optional[str] = None,
            overlap_secs: int = 2,
            loops: int = 12,
            start_step: int = 500
    ) -> Generator[np.ndarray, None, Tuple[int, np.ndarray]]:
        self.load_model(model_name)
        sample_rate, audio = sample_rate_audio
        # sample_rate = self.pipe.mel.get_sample_rate()
        overlap_samples = overlap_secs * sample_rate
        logging.info(f"Extending audio with model {self.model_name}, overlap_secs: {overlap_secs}, loops: {loops}, start_step: {start_step}, sample_rate: {sample_rate}, overlap samples: {overlap_samples}")
        track = audio
        for x in range(loops):
            output = self.pipe(
                raw_audio=audio[-overlap_samples:],
                start_step=start_step,
                mask_start_secs=overlap_secs,
            )
            audio = output.audios[0, 0]
            track = np.concatenate([track, audio[overlap_samples:]])
            yield audio
        return sample_rate, track

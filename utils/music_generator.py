from tempfile import NamedTemporaryFile
from typing import Optional

from audiocraft.models import musicgen, audiogen
from audiocraft.data.audio import audio_write
from utils import cuda_garbage_collection
from utils._interfaces import DisposableModel


def torch_optimizer(func):
    import torch
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast("cuda"):
                return func(*args, **kwargs)

    return wrapped


class MusicGenerator(DisposableModel):
    model = None
    model_name = None

    def __init__(self, model_name="facebook/musicgen-large", music: bool = True):
        self.model = None
        self.model_name = model_name
        self.music = music

    def load_model(self, model_name: Optional[str] = None, music: Optional[bool] = None):
        if model_name is not None:
            self.model_name = model_name
        if music is not None:
            self.music = music
        if not self.model or self.model.name != self.model_name:
            self.unload_model()
            if self.music:
                self.model = musicgen.MusicGen.get_pretrained(model_name)
            else:
                self.model = audiogen.AudioGen.get_pretrained(model_name)

    def unload_model(self):
        if self.model:
            del self.model
            cuda_garbage_collection()
            self.model = None


    @torch_optimizer
    def generate_music(
            self,
            prompt: str,
            model_name: Optional[str] = None,
            use_sampling: bool = True, top_k: int = 250,
            top_p: float = 0.0, temperature: float = 1.0,
            duration: float = 30.0, cfg_coef: float = 3.0,
            two_step_cfg: bool = False,
            extend_stride: float = 18,
            music: Optional[bool] = None
    ) -> str:
        self.load_model(model_name, music)
        self.model.set_generation_params(
            duration=duration,
            use_sampling=use_sampling,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
            two_step_cfg=two_step_cfg,
            extend_stride=extend_stride
        )
        outputs = self.model.generate([prompt])[0]
        outputs = outputs.detach().cpu().float()
        output = outputs[0]
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, self.model.sample_rate,
                strategy="loudness", format="wav",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
            )
            return file.name

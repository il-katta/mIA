from tempfile import NamedTemporaryFile

from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write
from utils import cuda_garbage_collection


class MusicGenerator(object):
    model = None
    model_name = None

    def __init__(self, model_name="facebook/musicgen-large"):
        self.model = None
        self.model_name = None

    def load_model(self, model_name=None):
        if model_name is not None:
            self.model_name = model_name
        if not self.model or self.model.name != self.model_name:
            del self.model
            cuda_garbage_collection()
            self.model = musicgen.MusicGen.get_pretrained(model_name)

    def generate_music(
            self,
            prompt: str,
            use_sampling: bool = True, top_k: int = 250,
            top_p: float = 0.0, temperature: float = 1.0,
            duration: float = 30.0, cfg_coef: float = 3.0,
            two_step_cfg: bool = False, extend_stride: float = 18
    ) -> str:
        if self.model is None:
            self.load_model()
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

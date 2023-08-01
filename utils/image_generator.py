import logging
from typing import Literal

from transformers import BitsAndBytesConfig

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from enum import Enum
import gc

DEFAULT_NEGATIVE_PROMPT = "3d, low quality, worst quality, glitch, glitch art, glitch effect, deformed, bad anatomy, bad perspective, bad composition, bad lighting, bad shadin"


class LoadMode(Enum):
    LOW_VRAM = 1
    LOAD_IN_4BIT = 2
    LOAD_IN_8BIT = 3
    GPU = 0


class ModelFamily(Enum):
    SD_15 = 1
    SD_2 = 2
    SD_XL_BASE = 4
    SD_XL_REFINER = 5


class ModelType(Enum):
    REALISTIC = "realistic"
    FANTASY = "fantasy"


MODELS = {
    "sd_xl_base": {
        "path": "/srv/data/stable-diffusion-webui/models/Stable-diffusion/sdxl/sd_xl_base_1.0.safetensors",
        "model_family": ModelFamily.SD_XL_BASE,
        "type": ModelType.REALISTIC,
    },
    "sd_xl_refiner": {
        "path": "/srv/data/stable-diffusion-webui/models/Stable-diffusion/sdxl/sd_xl_refiner_1.0.safetensors",
        "model_family": ModelFamily.SD_XL_REFINER,
        "type": ModelType.REALISTIC,
    },
    "photon": {
        "path": "/srv/data/stable-diffusion-webui/models/Stable-diffusion/photon_v1.safetensors",
        "model_family": ModelFamily.SD_15,
        "type": ModelType.REALISTIC,
    }
}
ModelNames = Literal["sd_xl_base", "sd_xl_refiner", "photon"]


class ImageGenerator(object):
    model_name: ModelNames
    model: dict
    pipe: StableDiffusionXLPipeline | StableDiffusionPipeline = None
    _cuda_pipe = None
    device: int

    def __init__(
            self,
            model_name: ModelNames,
            load_mode: LoadMode = LoadMode.GPU,
            device: int = 0
    ):
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.model_name = model_name
        self.model = MODELS[self.model_name]
        self.load_mode = load_mode

    def load_model(self):
        model_family = self.model["model_family"]
        safetensor_path = self.model["path"]
        self._logger.info(f"Loading model {self.model_name} from {safetensor_path}")

        image_size = 512
        if model_family == ModelFamily.SD_15:
            image_size = 512
        if model_family == ModelFamily.SD_2:
            image_size = 768
        if model_family == ModelFamily.SD_XL_BASE:
            image_size = 1024
        if model_family == ModelFamily.SD_XL_REFINER:
            image_size = 1024

        if self.load_mode == LoadMode.LOW_VRAM or self.load_mode == LoadMode.GPU:
            pipe: StableDiffusionXLPipeline | StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                safetensor_path,
                torch_dtype=torch.float16,
                image_size=image_size,
                use_safetensors=True,
                variant="fp16"
            )

            if self.load_mode == LoadMode.LOW_VRAM:
                pipe.enable_model_cpu_offload()
            else:
                self._cuda_pipe = pipe.to(f"cuda:{self.device}")

        elif self.load_mode == LoadMode.LOAD_IN_4BIT or self.load_mode == LoadMode.LOAD_IN_8BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_mode == LoadMode.LOAD_IN_8BIT,
                load_in_4bit=self.load_mode == LoadMode.LOAD_IN_4BIT,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            pipe: StableDiffusionXLPipeline | StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                safetensor_path,
                image_size=image_size,
                use_safetensors=True,
                quantization_config=quantization_config,
                device_map='auto',
            )
        else:
            raise ValueError(f"Invalid load mode {self.load_mode}")

        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        # if using torch < 2.0
        # pipe.enable_xformers_memory_efficient_attention()

        self.pipe = pipe

    def generate_image(
            self,
            subject: str,
            skip_sd_xl_refiner: bool = True,
            negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    ):
        if self.pipe is None:
            self.load_model()
        if self.model["model_family"] == ModelFamily.SD_XL_BASE:
            image = self.pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                output_type="latent" if not skip_sd_xl_refiner else "pil",
                height=1024,
                width=1024
            ).images[0]

            if not skip_sd_xl_refiner:
                image = self.sdxl_refiner(image, subject, negative_prompt)
        else:
            image = self.pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                output_type="pil",
            ).images[0]
        return image

    def sdxl_refiner(self, image, subject: str, negative_prompt: str = DEFAULT_NEGATIVE_PROMPT):
        original_model = self.model_name
        try:
            if self.model_name != "sd_xl_refiner":
                self.switch_model("sd_xl_refiner")
            image = self.pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                image=image,
                output_type="pil",
            ).images[0]
            return image
        finally:
            if self.model_name != original_model:
                self.switch_model(original_model)

    def unload_model(self):
        self._logger.debug(f"Unloading model {self.model_name}")
        if self.pipe is not None:
            if hasattr(self.pipe, "unet"):
                del self.pipe.unet
            del self.pipe

        if self._cuda_pipe is not None:
            if hasattr(self._cuda_pipe, "unet"):
                try:
                    del self._cuda_pipe.unet
                except AttributeError:
                    pass
            del self._cuda_pipe
        gc.collect()
        self.gc()

    def switch_model(self, new_model_name: ModelNames):
        self._logger.info(f"Switching model from {self.model_name} to {new_model_name}")
        self.unload_model()
        self.model_name = new_model_name
        self.model = MODELS[new_model_name]
        self.load_model()

    @staticmethod
    def gc():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def __del__(self):
        self.unload_model()

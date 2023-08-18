import gc
from typing import Optional
import PIL.Image
import math
from diffusers import StableDiffusionInstructPix2PixPipeline
import config
from utils import cuda_garbage_collection
from utils._interfaces import DisposableModel
import torch

from utils.image_generator import SCHEDULERS
from utils._torch_utils import torch_optimizer
import PIL.ImageOps


class Pix2Pix(DisposableModel):

    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix"):
        self._model_id = model_id
        self.pipe: Optional[StableDiffusionInstructPix2PixPipeline] = None

    @torch_optimizer
    def generate(self, img: PIL.Image, prompt: str, num_inference_steps=10, image_guidance_scale=1, resolution: int = 512) -> PIL.Image:
        self.load_model()
        width, height = img.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        img = PIL.ImageOps.fit(img, (width, height), method=PIL.Image.Resampling.LANCZOS)

        images = self.pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
        ).images
        return images[0]

    def unload_model(self):
        if self.pipe is None:
            if hasattr(self.pipe, "unet"):
                del self.pipe.unet
            if hasattr(self.pipe, "tokenizer"):
                del self.pipe.tokenizer
            if hasattr(self.pipe, "text_encoder"):
                del self.pipe.text_encoder
            if hasattr(self.pipe, "scheduler"):
                del self.pipe.scheduler
            del self.pipe
            self.pipe = None
            gc.collect()
            cuda_garbage_collection()

    def load_model(self, sampler_name="Euler a"):
        if self.pipe is not None:
            return

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=config.DATA_DIR / "huggingface"
        )
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        #self.pipe.unet.to(memory_format=torch.channels_last)
        #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        if sampler_name in SCHEDULERS.keys():
            self.pipe.scheduler = SCHEDULERS[sampler_name]["class"].from_config(
                self.pipe.config | SCHEDULERS[sampler_name]["config"]
            )

        #self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

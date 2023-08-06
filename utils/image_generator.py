import os
from datetime import datetime
import logging
import random
import sys
from typing import Literal, Optional

from PIL.PngImagePlugin import PngInfo
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import BitsAndBytesConfig
from diffusers.utils import get_class_from_dynamic_module

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, EulerDiscreteScheduler, \
    EulerAncestralDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, SchedulerMixin, \
    StableDiffusionXLImg2ImgPipeline, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, \
    DPMSolverSinglestepScheduler, \
    HeunDiscreteScheduler
import torch
from enum import Enum
import gc

import config
from utils import cuda_garbage_collection

DEFAULT_NEGATIVE_PROMPT = "portrait, 3d, low quality, worst quality, glitch, glitch art, glitch effect, deformed, bad anatomy, bad perspective, bad composition, bad lighting, bad shadin"

#TODO: look at stable-diffusion-webui/modules/txt2img.py
# see also https://github.com/vicgalle/stable-diffusion-aesthetic-gradients
# and https://github.com/CompVis/stable-diffusion

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
        "path": str(config.DATA_DIR / "stable-diffusion" / "models" / "sd_xl_base_1.0.safetensors"),
        "model_family": ModelFamily.SD_XL_BASE,
        "type": ModelType.REALISTIC,
    },
    "sd_xl_refiner": {
        "path": str(config.DATA_DIR / "stable-diffusion" / "models" / "sd_xl_refiner_1.0.safetensors"),
        "model_family": ModelFamily.SD_XL_REFINER,
        "type": ModelType.REALISTIC,
    },
    "photon": {
        "path": str(config.DATA_DIR / "stable-diffusion" / "models" / "photon_v1.safetensors"),
        "model_family": ModelFamily.SD_15,
        "type": ModelType.REALISTIC,
    },
    "rundiffusionFX": {
        "path": str(config.DATA_DIR / "stable-diffusion" / "models" / "rundiffusionFX_v10.safetensors"),
        "model_family": ModelFamily.SD_15,
        "type": ModelType.REALISTIC,
    },
    "rundiffusionFX25D": {
        "path": str(config.DATA_DIR / "stable-diffusion" / "models" / "rundiffusionFX25D_v10.safetensors"),
        "model_family": ModelFamily.SD_15,
        "type": ModelType.FANTASY,
    },
}
ModelNames = Literal["sd_xl_base", "sd_xl_refiner", "photon", "rundiffusionFX", "rundiffusionFX25D"]


# https://huggingface.co/docs/diffusers/v0.19.3/en/api/pipelines/stable_diffusion/stable_diffusion_xl
# https://huggingface.co/docs/diffusers/v0.19.3/en/api/pipelines/stable_diffusion/overview
# https://huggingface.co/docs/diffusers/v0.19.3/en/api/pipelines/stable_diffusion/text2img


class ImageGenerator(object):
    model_name: ModelNames
    model: dict
    pipe: StableDiffusionXLPipeline | StableDiffusionPipeline | StableDiffusionXLImg2ImgPipeline = None
    _default_scheduler: Optional[SchedulerMixin] = None
    device: int

    def __init__(
            self,
            model_name: ModelNames = "sd_xl_base",
            load_mode: LoadMode = LoadMode.GPU,
            nsfw_check: bool = False,
            use_lpw: bool = True,
            device: int = 0
    ):
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.model_name = model_name
        self.model = MODELS[self.model_name]
        self.load_mode = load_mode
        self.nsfw_check = nsfw_check
        self.use_lpw = use_lpw

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
            self.pipe = StableDiffusionPipeline.from_single_file(
                safetensor_path,
                torch_dtype=torch.float16,
                image_size=image_size,
                use_safetensors=True,
                safety_checker=StableDiffusionSafetyChecker if self.nsfw_check else None,
                requires_safety_checker=self.nsfw_check,
                variant="bf16"
                # variant="fp16"
            ).to(f"cuda:{self.device}")

            if self.load_mode == LoadMode.LOW_VRAM:
                self.pipe.enable_model_cpu_offload(gpu_id=self.device)

        elif self.load_mode == LoadMode.LOAD_IN_4BIT or self.load_mode == LoadMode.LOAD_IN_8BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_mode == LoadMode.LOAD_IN_8BIT,
                load_in_4bit=self.load_mode == LoadMode.LOAD_IN_4BIT,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.pipe = StableDiffusionPipeline.from_single_file(
                safetensor_path,
                image_size=image_size,
                use_safetensors=True,
                quantization_config=quantization_config,
                device_map='auto',
                # safety_checker=None if not self.nsfw_check else StableDiffusionSafetyChecker,
                # requires_safety_checker=self.nsfw_check,
            )
        else:
            raise ValueError(f"Invalid load mode {self.load_mode}")

        if self.use_lpw:
            LPWStableDiffusionPipeline = get_class_from_dynamic_module("lpw_stable_diffusion",
                                                                       module_file="lpw_stable_diffusion.py")
            self.pipe = LPWStableDiffusionPipeline(**self.pipe.components)

        self.pipe.unet.to(memory_format=torch.channels_last)
        if not self.use_lpw:  # compilation is not compatible with lpw
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        ## if using torch < 2.0
        ## pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()
        # self.pipe.enable_model_cpu_offload(gpu_id=self.device)

        self.pipe.safety_checker = None

        self._default_scheduler = self.pipe.scheduler

    def generate_image(
            self,
            subject: str,
            skip_sd_xl_refiner: bool = True,
            negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
            sampler_name: Optional[str] = None,
            num_inference_steps=32,
            guidance_scale=7,
            seed: int = -1
    ):
        if self.pipe is None:
            self.load_model()
        if seed < 1:
            seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(self.pipe.device).manual_seed(seed)

        self._logger.info(f"Using sampler {sampler_name}")
        if sampler_name == "DDIM":
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "Euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "Euler a":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "DPM2 Karras":
            self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "DPM2 a Karras":
            self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "LMS":
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "DPM Solver":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config | {"algorithm_type": "dpmsolver"})
        elif sampler_name == "DPM Solver++":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config | {"algorithm_type": "dpmsolver++"})
        elif sampler_name == "DPM Single":
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "heun":
            self.pipe.scheduler = HeunDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = self._default_scheduler

        if self.model["model_family"] == ModelFamily.SD_XL_BASE:
            image = self.pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                output_type="latent" if not skip_sd_xl_refiner else "pil",
                height=1024,
                width=1024,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_embeddings_multiples=5,
            ).images[0]

            if not skip_sd_xl_refiner:
                image = self.sdxl_refiner(image, subject, negative_prompt)
        else:
            image = self.pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                output_type="pil",
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_embeddings_multiples=5,
            ).images[0]
        generation_params = {
            "Steps": num_inference_steps,
            "Sampler": self.pipe.scheduler.__class__.__name__,
            "CFG scale": guidance_scale,
            "Seed": seed,
            # "Size": f"{width}x{height}",
            "Model": self.model_name,
        }

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
        prompt_text = subject if subject else ""
        negative_prompt_text = "Negative prompt: " + negative_prompt if negative_prompt else ""
        metadata = PngInfo()
        metadata.add_text("parameters", f"{prompt_text}\n{negative_prompt_text}\n{generation_params_text}".strip())
        outputdir = config.DATA_DIR / "stable-diffusion" / "outputs"
        os.makedirs(str(outputdir), exist_ok=True)
        image.save(str(outputdir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"), pnginfo=metadata)
        return image, metadata

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
        cuda_garbage_collection()

    def __del__(self):
        self.unload_model()

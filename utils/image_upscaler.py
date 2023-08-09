from typing import Optional

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
import torch
import gc

from transformers import BitsAndBytesConfig

from utils import cuda_garbage_collection
from utils._interfaces import DisposableModel


class ImageUpscaler(DisposableModel):
    _upscaler: Optional[StableDiffusionUpscalePipeline] = None

    MODELS = {
        "stabilityai/stable-diffusion-x4-upscaler": StableDiffusionUpscalePipeline,
        "stabilityai/sd-x2-latent-upscaler": StableDiffusionLatentUpscalePipeline
    }

    def __init__(self, model="stabilityai/stable-diffusion-x4-upscaler"):
        '''

        :param model: suggested: "stabilityai/stable-diffusion-x4-upscaler" "stabilityai/sd-x2-latent-upscaler"
        '''
        self.model = model
        self.Pipeline = self.MODELS[model]

    def upscale(
            self,
            image: Image.Image,
            prompt: Optional[str] = None,
            negative_prompt: Optional[str] = None,
            num_inference_steps=75,
            guidance_scale=9,
            noise_level=20
    ):
        if self._upscaler is None:
            self.load_model()
        upscaled_image = self._upscaler(prompt=prompt, image=image, negative_prompt=negative_prompt,
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                        # noise_level=noise_level,
                                        output_type="pil").images[0]
        return upscaled_image

    def load_model(
            self,
            enable_attention_slicing=True,
            enable_model_cpu_offload=True,
            enable_xformers_memory_efficient_attention=True,
            load_in_4bit=False,
            load_in_8bit=False,
            compile_unet=False
    ):
        if self._upscaler is None:
            if load_in_8bit or load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                device_map = "auto"
            else:
                quantization_config = None
                device_map = None
            self._upscaler = self.Pipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                device_map=device_map,
                quantization_config=quantization_config
            )
            if not load_in_8bit and not load_in_4bit:
                self._upscaler = self._upscaler.to("cuda")
            self._upscaler.unet.to(memory_format=torch.channels_last)
            if compile_unet:
                self._upscaler.unet = torch.compile(self._upscaler.unet, mode="reduce-overhead", fullgraph=True)
            if enable_attention_slicing:
                self._upscaler.enable_attention_slicing()
            if enable_model_cpu_offload:
                self._upscaler.enable_model_cpu_offload()
            if enable_xformers_memory_efficient_attention:
                self._upscaler.enable_xformers_memory_efficient_attention()

    def unload_model(self):
        if self._upscaler is not None:
            del self._upscaler
            self._upscaler = None

        cuda_garbage_collection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()

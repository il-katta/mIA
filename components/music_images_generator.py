import json

import gradio as gr
from transformers import BitsAndBytesConfig

import config
from bot import MiaBot
import openai
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from pathlib import Path
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


def load_model(
        safetensor_path: str | Path,
        model_family: ModelFamily,
        load_mode: LoadMode = LoadMode.GPU
) -> StableDiffusionXLPipeline | StableDiffusionPipeline:
    image_size = 512

    if model_family == ModelFamily.SD_15:
        image_size = 512
    if model_family == ModelFamily.SD_2:
        image_size = 768
    if model_family == ModelFamily.SD_XL_BASE:
        image_size = 1024
    if model_family == ModelFamily.SD_XL_REFINER:
        image_size = 1024

    if load_mode == LoadMode.LOW_VRAM or load_mode == LoadMode.GPU:
        pipe: StableDiffusionXLPipeline | StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
            safetensor_path,
            torch_dtype=torch.float16,
            image_size=image_size,
            use_safetensors=True,
            variant="fp16"
        )

        if load_mode == LoadMode.LOW_VRAM:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")

    elif load_mode == LoadMode.LOAD_IN_4BIT or load_mode == LoadMode.LOAD_IN_8BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_mode == LoadMode.LOAD_IN_8BIT,
            load_in_4bit=load_mode == LoadMode.LOAD_IN_4BIT,
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
        raise ValueError(f"Invalid load mode {load_mode}")

    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    gc.collect()
    return pipe


def unload_pipe():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def generate_image(
        subject: str,
        model_name: str,
        load_mode: LoadMode = LoadMode.GPU,
        skip_sd_xl_refiner: bool = False,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
):
    model = MODELS[model_name]

    pipe = load_model(safetensor_path=model["path"], model_family=model["model_family"], load_mode=load_mode)

    if model["model_family"] == ModelFamily.SD_XL_BASE:
        image = pipe(
            prompt=subject,
            negative_prompt=negative_prompt,
            output_type="latent" if not skip_sd_xl_refiner else "pil",
            height=1024,
            width=1024
        ).images[0]
        del pipe.unet
        del pipe
        gc.collect()
        unload_pipe()
        if not skip_sd_xl_refiner:
            model_refiner = MODELS["sd_xl_refiner"]
            pipe = load_model(
                safetensor_path=model_refiner["path"],
                model_family=model_refiner["model_family"],
                load_mode=load_mode
            )
            image = pipe(
                prompt=subject,
                negative_prompt=negative_prompt,
                image=image,
                output_type="pil",
            ).images[0]
            del pipe.unet
            del pipe
    else:
        image = pipe(
            prompt=subject,
            negative_prompt=negative_prompt,
            output_type="pil",
        ).images[0]
        del pipe.unet
        del pipe
    gc.collect()
    unload_pipe()
    return image


def call_openai_api(text: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": f"Generate image for the song {text}"
            }
        ],
        functions=[
            {
                "name": "generate_image",
                "description": """
      Generate an image based on the description of the elements present in the image and the type of image you want to generate.
      the description of the image are a list of details that must be present in the image representative of the song lyrics or the song meaning. 
      The details must be concrete, devoid of abstract concepts, for example instead of 'A corrupted city' the correct parameter is 'A city, garbage on the street, burning cars'. 
      The image ca be realistic or fantasy, you will also need to specify if the image is a photo therefore a real thing or a drawing because it includes elements of fantasy.
      """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "the description of the elements present in the image, separated by comma. For example: 'a man, a dog, a tree, night, moon'",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["realistic", "fantasy"],
                            "description": "the type of image you want to generate, 'realistic' if the image should be realistic, 'fantasy' if the image should be fantasy"
                        }
                    },
                    "required": ["subject", "type"]
                }
            }
        ],
        temperature=1,
    )
    for choice in response.choices:
        if choice.message.get("function_call", None):
            function_name = choice.message["function_call"]["name"]
            args = json.loads(choice.message["function_call"]["arguments"])
            yield function_name, args
        try:
            args = json.loads(choice.message["content"])
            yield "generate_image", args
        except:
            pass


def gui(bot: MiaBot, conf: config.Config):
    pass

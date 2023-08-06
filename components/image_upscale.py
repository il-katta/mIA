from typing import Optional

import gradio as gr

from utils import package_exists


def is_available():
    return package_exists("torch") and package_exists("transformers") and package_exists("diffusers")


def gui():
    from utils import cuda_garbage_collection
    from utils.image_upscaler import ImageUpscaler

    with gr.Row():
        image_in = gr.Image(label="Input Image", type="pil", image_mode="RGB")

    model_choose = gr.Radio(
        choices=[
            "stabilityai/sd-x2-latent-upscaler",
            "stabilityai/stable-diffusion-x4-upscaler"
        ],
        label="Model",
        value="stabilityai/sd-x2-latent-upscaler"
    )
    prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Prompt")
    negative_prompt = gr.Textbox(label="Negative Prompt", lines=1, placeholder="Negative Prompt")

    num_inference_steps_number = gr.Number(label="Num inference steps", value=75, precision=0, minimum=1)
    guidance_scale_number = gr.Number(label="Guidance scale", value=9, precision=0, minimum=1)
    noise_level_number = gr.Number(label="Noise level", value=20, precision=0, minimum=1)

    with gr.Row():
        enable_attention_slicing_checkbox = gr.Checkbox(label="Enable attention slicing", value=True)
        enable_model_cpu_offload_checkbox = gr.Checkbox(label="Enable model CPU offload", value=True)
        enable_xformers_memory_efficient_attention_checkbox = gr.Checkbox(
            label="Enable xformers memory efficient attention", value=True)
        load_in_4bit_checkbox = gr.Checkbox(label="Load in 4bit", value=False)
        load_in_8bit_checkbox = gr.Checkbox(label="Load in 8bit", value=False)

    with gr.Row():
        execute_button = gr.Button("Execute")
        free_vram_button = gr.Button("Free VRAM")

    with gr.Row():
        image_out = gr.Image(label="Output Image", type="pil", image_mode="RGB", interactive=False)

    def upscale(
            image,
            model_name: str,
            prompt: Optional[str] = None,
            negative_prompt: Optional[str] = None,
            num_inference_steps=75,
            guidance_scale=9,
            noise_level=20,
            enable_attention_slicing=True,
            enable_model_cpu_offload=True,
            enable_xformers_memory_efficient_attention=True,
            load_in_4bit=False,
            load_in_8bit=False
    ):
        try:
            with ImageUpscaler(model_name) as image_upscaler:
                image_upscaler.load_model(
                    enable_attention_slicing=enable_attention_slicing,
                    enable_model_cpu_offload=enable_model_cpu_offload,
                    enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                image = image_upscaler.upscale(
                    image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level
                )

            return image
        finally:
            cuda_garbage_collection()

    execute_button.click(
        upscale,
        inputs=[
            image_in,
            model_choose,
            prompt,
            negative_prompt,
            num_inference_steps_number,
            guidance_scale_number,
            noise_level_number,
            enable_attention_slicing_checkbox,
            enable_model_cpu_offload_checkbox,
            enable_xformers_memory_efficient_attention_checkbox,
            load_in_4bit_checkbox,
            load_in_8bit_checkbox
        ],
        outputs=[image_out],
    )

    free_vram_button.click(cuda_garbage_collection)

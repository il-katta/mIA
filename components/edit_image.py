import gradio as gr

from utils import package_exists, cuda_is_available
from utils.system_stats import SystemStats


def is_available():
    return package_exists("diffusers") and package_exists("torch") and cuda_is_available()


def gui(sysstats: SystemStats):
    from utils.pix2pix import Pix2Pix
    pix2pix = Pix2Pix()
    sysstats.register_disposable_model(pix2pix)

    with gr.Row():
        source_image = gr.Image(type="pil", image_mode="RGB", label="Source Image")
    with gr.Row():
        prompt_text = gr.Textbox(lines=5, label="Prompt")
    with gr.Row():
        num_inference_steps_slider = gr.Slider(minimum=1, maximum=500, value=10, step=1, label="Inference Steps")
        image_guidance_scale_slider = gr.Slider(minimum=0, maximum=200, value=1, step=0.5, label="Image Guidance Scale")

    with gr.Row():
        submit_button = gr.Button("↗️", variant="primary")

    with gr.Row():
        output_image = gr.Image(type="pil", image_mode="RGB", label="Output Image")

    def generate_image(image, prompt: str, num_inference_steps: int, image_guidance_scale: float):
        output_image = pix2pix.generate(image, prompt, num_inference_steps, image_guidance_scale)
        return output_image

    submit_button.click(
        generate_image,
        inputs=[source_image, prompt_text, num_inference_steps_slider, image_guidance_scale_slider],
        outputs=[output_image]
    )

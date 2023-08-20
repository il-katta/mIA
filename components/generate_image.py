from utils import package_exists, cuda_is_available
from utils.system_stats import SystemStats
import gradio as gr
import PIL.Image


def is_available():
    return package_exists("diffusers") and package_exists("torch") and cuda_is_available()


def gui(sysstats: SystemStats):
    from utils.image_generator import ImageGenerator, MODELS, SCHEDULERS, DEFAULT_NEGATIVE_PROMPT

    image_generator = ImageGenerator()
    sysstats.register_disposable_model(image_generator)
    with gr.Row():
        model_name_select = gr.Dropdown(list(MODELS.keys()), label="Model Name", value=list(MODELS.keys())[0])
    with gr.Row():
        prompt_text = gr.Textbox(lines=5, label="Prompt")
    with gr.Row():
        negative_prompt_text = gr.Textbox(lines=5, label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT)

    with gr.Row():
        sampler_name_select = gr.Dropdown(
            list(SCHEDULERS.keys()),
            label="Sampler Name",
        )
        num_inference_steps = gr.Slider(minimum=1, maximum=500, value=32, step=1, label="Inference Steps")
        image_guidance_scale = gr.Slider(minimum=0, maximum=200, value=7, step=0.5,
                                         label="Image Guidance Scale")

        seed_number = gr.Number(value=-1, label="Seed", precision=0)

    with gr.Row():
        submit_button = gr.Button("↗️", variant="primary")

    with gr.Row():
        image_viewer = gr.Image(type="pil", image_mode="RGB")

    def generate_image(
            model_name, prompt_text: str, negative_prompt_text: str, sampler_name: str, num_inference_steps: float,
            image_guidance_scale: float, seed: int
    ) -> 'PIL.Image':
        if image_generator.model_name != model_name:
            image_generator.switch_model(model_name)
        if seed < 0:
            seed = image_generator.generate_seed()
        image = image_generator.generate_image(
            subject=prompt_text,
            negative_prompt=negative_prompt_text,
            sampler_name=sampler_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=image_guidance_scale,
            seed=seed
        )
        return image[0], seed

    def on_model_name_select_change(model_name: str) -> str:
        image_generator.switch_model(model_name)
        return model_name

    submit_button.click(
        generate_image,
        inputs=[
            model_name_select, prompt_text, negative_prompt_text, sampler_name_select, num_inference_steps,
            image_guidance_scale, seed_number
        ],
        outputs=[image_viewer, seed_number],
        api_name="generate_image"
    )

    model_name_select.change(
        on_model_name_select_change,
        inputs=[model_name_select],
        outputs=[model_name_select],
        api_name="load_model"
    )

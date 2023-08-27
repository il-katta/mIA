from typing import List

import gradio as gr

from utils import package_exists
from utils.system_stats import SystemStats

banned_tags = ["unused"]

tag_translations = {
    "rating:safe": "sfw",
}
DEFAULT_MODEL_LOAD_IN_4BIT = True


def is_available():
    return package_exists("onnxruntime")


def gui(sysstats: SystemStats):
    from utils.image_captioning import ImageCaptioning

    image_captioning = ImageCaptioning(DEFAULT_MODEL_LOAD_IN_4BIT)
    sysstats.register_disposable_model(image_captioning)

    def generate_model_list(load_in_4bit: bool = DEFAULT_MODEL_LOAD_IN_4BIT):
        image_captioning.init_models(load_in_4bit=load_in_4bit)

    def predict_single(image, models_choose: List[str]):
        predictions = image_captioning.predict_image(
            image,
            models_choose,
            # don't know why, it's needed to call this function too in order to free the vram
            lambda x: sysstats.free_vram()
        )
        return ", ".join(list(predictions))

    def prealod_models(models_choose: List[str]):
        image_captioning.load_models(models_choose)
        return gr.Button.update()

    def tag_directory(
            input_directory_path: str,
            output_directory_path: str,
            models_choose: List[str],
            resize: bool = True,
            resize_size: int = 512,
            overwrite: bool = False,
            keep_existing_tags: bool = True,
            additional_tags_prefix: str = "",
            additional_tags_postfix: str = "",
    ):
        image_captioning.tag_directory(
            input_directory_path=input_directory_path,
            output_directory_path=output_directory_path,
            models_choose=models_choose,
            resize=resize,
            resize_size=resize_size,
            overwrite=overwrite,
            keep_existing_tags=keep_existing_tags,
            additional_tags_prefix=additional_tags_prefix,
            additional_tags_postfix=additional_tags_postfix,
            # don't know why, it's needed to call this function too in order to free the vram
            callback_after_model_unload=lambda x: sysstats.free_vram()
        )

        return gr.Button.update()

    generate_model_list()

    with gr.Row():
        models_selection = gr.Checkboxgroup(
            label="Models",
            choices=image_captioning.model_names,
            value=image_captioning.model_names,
            type="value",
        )

    with gr.Row():
        models_load_in_4bit_checkbox = gr.Checkbox(
            label="Load in 4bit",
            value=DEFAULT_MODEL_LOAD_IN_4BIT,
        )
        models_load_in_4bit_checkbox.change(
            generate_model_list,
            inputs=[models_load_in_4bit_checkbox],
        )

        preload_button = gr.Button(value="Preload models")
    with gr.Tab("Single image"):
        with gr.Row():
            image_in = gr.Image(label="Input Image", type="pil", image_mode="RGB")

        with gr.Row():
            tags_out = gr.Label(text="Tags", label="Output Tags", interactive=False)

        with gr.Row():
            single_image_button = gr.Button(value="Run")

    with gr.Tab("Tag directory"):
        with gr.Row():
            directory_input_text = gr.Textbox(label="Input directory")
            directory_output_text = gr.Textbox(label="Output directory")

        with gr.Row():
            override_checkbox = gr.Checkbox(label="Override existing images in output directory", value=False)
            keep_existing_tags_checkbox = gr.Checkbox(label="Keep existing tags in target directory", value=True)
            resize_checkbox = gr.Checkbox(label="Resize images", value=True)
            resize_size_number = gr.Slider(label="Resize size", value=512, minimum=1, maximum=4096, step=1)

        with gr.Row():
            additional_tags_prefix_text = gr.Textbox(label="Additional tags prefix")
            additional_tags_postfix_text = gr.Textbox(label="Additional tags postfix")

        with gr.Row():
            directory_path_button = gr.Button(value="Run")

    preload_button.click(prealod_models, inputs=[models_selection], outputs=[preload_button])
    single_image_button.click(predict_single, [image_in, models_selection], [tags_out])

    directory_path_button.click(
        tag_directory,
        inputs=[
            directory_input_text,
            directory_output_text,
            models_selection,
            resize_checkbox,
            resize_size_number,
            override_checkbox,
            keep_existing_tags_checkbox,
            additional_tags_prefix_text,
            additional_tags_postfix_text,
        ],
        outputs=[directory_path_button]
    )

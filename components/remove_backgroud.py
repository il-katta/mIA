import gradio as gr

from utils import package_exists


def is_available():
    return package_exists("rembg")


def gui():
    import rembg
    import rembg.sessions
    with gr.Row():
        image_in = gr.Image(label="Input Image")

    model_choose = gr.Radio(choices=rembg.sessions.sessions_names, label="Model", value="isnet-general-use")
    alpha_matting_checkbox = gr.Checkbox(label="Alpha Matting", value=False)
    alpha_matting_foreground_threshold = gr.Slider(label="Foreground Threshold", minimum=0, maximum=255, step=1,
                                                   value=240)
    alpha_matting_background_threshold = gr.Number(label="Background Threshold", minimum=0, maximum=255, step=1,
                                                   value=10)
    alpha_matting_erode_size = gr.Number(label="Erode Size", value=10, step=1, precision=0)
    only_mask_checkbox = gr.Checkbox(label="Only Mask", value=False)
    post_process_mask_checkbox = gr.Checkbox(label="Post Process Mask", value=False)

    with gr.Row():
        bgcolor = gr.ColorPicker(label="Background Color", value="#000000")
        bgcolor_alpha = gr.Slider(label="Background Alpha", value=0, step=1, minimum=0, maximum=255)

    with gr.Row():
        execute_button = gr.Button("Execute")

    with gr.Row():
        image_out = gr.Image(label="Output Image", interactive=False)

    def remove(
            image,
            model_name: str,
            alpha_matting: bool,
            alpha_matting_foreground_threshold: int,
            alpha_matting_background_threshold: int,
            alpha_matting_erode_size: int,
            only_mask: bool,
            post_process_mask: bool,
            bgcolor_hex: str,
            bgcolor_alpha: int
    ):
        bgcolor = (int(bgcolor_hex[1:3], 16), int(bgcolor_hex[3:5], 16), int(bgcolor_hex[5:7], 16), bgcolor_alpha)
        return rembg.remove(
            image,
            alpha_matting=alpha_matting,
            session=rembg.new_session(model_name),
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            bgcolor=bgcolor
        )

    execute_button.click(
        remove,
        inputs=[image_in, model_choose, alpha_matting_checkbox, alpha_matting_foreground_threshold,
                alpha_matting_background_threshold, alpha_matting_erode_size, only_mask_checkbox,
                post_process_mask_checkbox, bgcolor, bgcolor_alpha],
        outputs=[image_out],
    )

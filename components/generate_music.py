import gradio as gr

from utils import package_exists


def is_available():
    return package_exists("torch") and package_exists("audiocraft")


def gui():
    from utils.music_generator import MusicGenerator
    music_generator = MusicGenerator()

    def generate_music(prompt, model_name, duration, top_k, top_p, temperature, cfg_coef):
        music_generator.load_model(model_name)
        return music_generator.generate_music(
            prompt, duration=duration, top_k=top_k, top_p=top_p, temperature=temperature, cfg_coef=cfg_coef
        )

    with gr.Row():
        prompt_text = gr.Text(label="Input Text", interactive=True)

    with gr.Row():
        model_radio = gr.Radio(
            ["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
             "facebook/musicgen-large"],
            label="Model", value="facebook/musicgen-melody", interactive=True
        )
    with gr.Row():
        duration_slider = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
    with gr.Row():
        topk_number = gr.Number(label="Top-k", value=250, interactive=True)
        topp_number = gr.Number(label="Top-p", value=0, interactive=True)
        temperature_number = gr.Number(label="Temperature", value=1.0, interactive=True)
        cfg_coef_number = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
    with gr.Row():
        submit_button = gr.Button("Generate")
    with gr.Row():
        audio_output = gr.Audio(label="Generated Music", type='filepath')

    submit_button.click(
        generate_music,
        inputs=[
            prompt_text, model_radio, duration_slider, topk_number, topp_number, temperature_number, cfg_coef_number
        ],
        outputs=[audio_output]
    )

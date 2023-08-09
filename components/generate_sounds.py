import gradio as gr

from utils import package_exists
from utils.system_stats import SystemStats


def is_available():
    return package_exists("torch") and package_exists("audiocraft")


def gui(sysstats: SystemStats):
    from utils.music_generator import MusicGenerator
    music_generator = MusicGenerator(music=False)
    sysstats.register_disposable_model(music_generator)

    def generate_sound(prompt, model_name, duration, top_k, top_p, temperature, cfg_coef, extend_stride):
        return music_generator.generate_music(
            prompt,
            model_name=model_name,
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
            extend_stride=extend_stride,
            music=False,
        )

    with gr.Row():
        prompt_text = gr.Text(
            label="Prompt",
            placeholder="For example: 'dog barking' or 'sirene of an emergency vehicle'",
            interactive=True
        )

    with gr.Row():
        model_radio = gr.Radio(
            [
                "facebook/audiogen-medium",
             ],
            label="Model", value="facebook/audiogen-medium", interactive=True
        )
    with gr.Row():
        duration_slider = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
    with gr.Row():
        topk_number = gr.Number(label="Top-k", value=250, interactive=True, precision=0)
        topp_number = gr.Number(label="Top-p", value=0, interactive=True, precision=0)
        temperature_number = gr.Number(label="Temperature", value=1.0, interactive=True)
        cfg_coef_number = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
        extend_stride_number = gr.Number(label="Extend stride", value=2.0, minimum=0, maximum=9, interactive=True)
    with gr.Row():
        submit_button = gr.Button("Generate")
    with gr.Row():
        audio_output = gr.Audio(label="Generated sound", type='filepath', autoplay=True)

    model_radio.change(
        music_generator.load_model,
        inputs=[model_radio],
        outputs=[],
    )

    submit_button.click(
        generate_sound,
        inputs=[
            prompt_text, model_radio, duration_slider, topk_number, topp_number, temperature_number, cfg_coef_number, extend_stride_number
        ],
        outputs=[audio_output]
    )

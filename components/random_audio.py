import gradio as gr

from utils import package_exists, cuda_is_available
from utils.system_stats import SystemStats


def is_available():
    return package_exists("torch") and package_exists("audiodiffusion") and cuda_is_available()


def gui(sysstats: SystemStats):
    from utils.audio_generator import AudioGenerator, MODELS
    audio_generator = AudioGenerator()
    sysstats.register_disposable_model(audio_generator)

    def generate_audio(model_name, loops, steps, seed, start_step):
        loop, audio = audio_generator.generate_audio(
            model_name=model_name,
            loops=loops,
            steps=steps,
            seed=seed,
            start_step=start_step
        )
        return loop, audio

    def load_model(model_name):
        audio_generator.load_model(model_name)
        return audio_generator.model_name

    with gr.Row():
        model_radio = gr.Radio(
            MODELS,
            label="Model", value=audio_generator.model_name, interactive=True
        )

    with gr.Group():
        with gr.Row():
            loops_number = gr.Number(label="Loops", value=1, interactive=True, precision=0)
            steps_number = gr.Number(label="Steps", value=1000, interactive=True, precision=0)
            seed_number = gr.Number(label="Seed", value=-1, interactive=True, precision=0)
            start_step_number = gr.Number(label="Start Step", value=0, interactive=True, precision=0)

        with gr.Row():
            submit_button = gr.Button("Generate")

        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", type='numpy', autoplay=False, interactive=False)
            loop_output = gr.Audio(label="Looped Audio", type='numpy', autoplay=True, interactive=False)

        model_radio.change(
            load_model,
            inputs=[model_radio],
            outputs=[model_radio]
        )

        submit_button.click(
            generate_audio,
            inputs=[model_radio, loops_number, steps_number, seed_number, start_step_number],
            outputs=[loop_output, audio_output],
        )

    with gr.Group():
        with gr.Row():
            overlap_secs_number = gr.Number(label="Overlap Seconds", value=2, interactive=True, precision=0)
            repeat_loops_number = gr.Number(label="Loops", value=12, interactive=True, precision=0)
            ext_start_step_number = gr.Number(label="Start Step", value=500, interactive=True, precision=0)

        with gr.Row():
            extend_audio_button = gr.Button("Extend Audio")

        with gr.Row():
            extended_output = gr.Audio(label="Extended Audio", type='numpy', autoplay=True, interactive=False)

        def extend_audio(sample_rate_audio, model_name, overlap_secs, loops, start_step):
            return audio_generator.extend_audio(sample_rate_audio, model_name, overlap_secs, loops, start_step)

        extend_audio_button.click(
            extend_audio,
            inputs=[audio_output, model_radio, overlap_secs_number, repeat_loops_number, ext_start_step_number],
            outputs=[extended_output],
        )

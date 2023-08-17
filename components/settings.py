import os
import shutil
from typing import Tuple, Dict, Optional

import gradio as gr

import config
from utils.system_stats import SystemStats
from utils.tts import TextToSpeech


def gui(conf: config.Config, sysstats: SystemStats):
    tts = TextToSpeech(config.ELEVENLABS_DEFAULT_APIKEY)
    sysstats.register_disposable_model(tts)

    # Openai
    with gr.Row():
        openai_temperature_slider = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=conf.openai_temperature_state.value,
            label="Temperature",
        )
        openai_temperature_slider.change(
            lambda value: value,
            inputs=[openai_temperature_slider],
            outputs=[conf.openai_temperature_state]
        )

        openai_model_input = gr.Text(
            type="text",
            value=conf.openai_model_state.value,
            label="OpenAI Model",
        )

        openai_model_input.change(
            lambda value: value,
            inputs=[openai_model_input],
            outputs=[conf.openai_model_state]
        )

    # TTS Generator
    with gr.Row():
        tts_generator_radio = gr.Radio(
            choices=[config.GENERATOR_DISABLED, config.GENERATOR_ELEVENLABS, config.GENERATOR_BARK],
            value=conf.tts_generator_state.value,
            type="value",
            label="TTS Generator"
        )
        tts_generator_radio.change(
            lambda value: value,
            inputs=[tts_generator_radio],
            outputs=[conf.tts_generator_state]
        )

    def elevelabs_enabled(tts_generator: Optional[str] = None) -> bool:
        if tts_generator is None:
            tts_generator = conf.tts_generator_state.value
        return tts_generator == config.GENERATOR_ELEVENLABS

    def bark_enabled(tts_generator: Optional[str] = None) -> bool:
        if tts_generator is None:
            tts_generator = conf.tts_generator_state.value
        return tts_generator == config.GENERATOR_BARK

    # Bark
    with gr.Group(visible=bark_enabled()) as bark_row:
        with gr.Row():
            bark_voices = tts.bark_voices()
            bark_voices_radio = gr.Dropdown(
                choices=bark_voices,
                type="value",
                value=conf.bark_voice_id_state.value,
                label="Bark Voice",
            )
            bark_voices_radio.change(
                lambda value: value,
                inputs=[bark_voices_radio],
                outputs=[conf.bark_voice_id_state]
            )

            bark_device_radio = gr.Dropdown(
                choices=config.BARK_DEVICES,
                type="value",
                value=conf.bark_device_state.value,
                label="Bark Device",
            )

            bark_device_radio.change(
                lambda value: value,
                inputs=[bark_device_radio],
                outputs=[conf.bark_device_state]
            )

    # ElevenLabs
    with gr.Group(visible=elevelabs_enabled()) as elevenlabs_row:
        with gr.Row():
            elevenlabs_voices_values = tts.elevenlabs_voices()
            elevenlabs_voices_radio = gr.Dropdown(
                choices=list(elevenlabs_voices_values.values()),
                type="value",
                value=elevenlabs_voices_values.get(conf.elevenlabs_voice_id_state.value, None),
                label="ElevenLabs Voice",
            )
        with gr.Row():
            refresh_voices_btn = gr.Button("🔄")

        def elevenlabs_voices_radio_change(value: str) -> str:
            return next(key for (key, val) in elevenlabs_voices_values.items() if val == value)

        def elevenlabs_force_refresh() -> Dict[str, str]:
            return tts.elevenlabs_voices(force_fresh=True)

        elevenlabs_voices_radio.change(
            elevenlabs_voices_radio_change,
            inputs=[elevenlabs_voices_radio],
            outputs=[conf.elevenlabs_voice_id_state]
        )

        refresh_voices_btn.click(
            elevenlabs_force_refresh,
            inputs=None,
            outputs=[elevenlabs_voices_radio],
            queue=False
        )
    # test voice
    with gr.Group(visible=bark_enabled() or elevelabs_enabled()) as tts_test_row:
        with gr.Row():
            bark_test_btn = gr.Button("Test voice")

        with gr.Row():
            bark_test_audio = gr.Audio(
                container=False,
                autoplay=True,
            )

    def test_voice(status: str, elevenlabs_voice: str, bark_voice_id: str, bark_device: str) -> str:
        if status == config.GENERATOR_ELEVENLABS:
            elevenlabs_voice_id = elevenlabs_voices_radio_change(elevenlabs_voice)
            examples_dirpath = config.DATA_DIR / "elevenlabs"
            os.makedirs(examples_dirpath, exist_ok=True)
            examples_filepath = examples_dirpath / f"{elevenlabs_voice_id}.mp3"
            if not os.path.exists(examples_filepath):
                filepath = tts.elevenlabs_generate(
                    "Ciao. Come va? Sono un'intelligenza artificiale, un sistema avanzato progettato per interagire con gli utenti.",
                    elevenlabs_voice_id
                )
                shutil.move(filepath, examples_filepath)
            return examples_filepath
        elif status == config.GENERATOR_BARK:
            filepath = tts.bark_generate(
                "Ciao. Come va? [laughs] Sono un'intelligenza artificiale, un sistema avanzato progettato per interagire con gli utenti.",
                bark_voice_id,
                device=bark_device
            )
            return filepath
        else:
            return ""

    bark_test_btn.click(
        test_voice,
        inputs=[tts_generator_radio, elevenlabs_voices_radio, bark_voices_radio, bark_device_radio],
        outputs=[bark_test_audio],
        queue=False
    )

    def tts_generator_radio_change(value: str) -> Tuple[str, Dict, Dict, Dict]:
        return value, \
            gr.Radio.update(visible=elevelabs_enabled(value)), \
            gr.Radio.update(visible=bark_enabled(value)), \
            gr.Group.update(visible=value != config.GENERATOR_DISABLED)

    tts_generator_radio.change(
        tts_generator_radio_change,
        inputs=[tts_generator_radio],
        outputs=[conf.tts_generator_state, elevenlabs_row, bark_row, tts_test_row]
    )

    def save_config(
            openai_model_state: str,
            openai_temperature: int,
            tts_generator: str,
            elevenlabs_voice_id: str,
            bark_voice_id: str,
    ):
        conf.update(
            openai_model=openai_model_state,
            openai_temperature=openai_temperature,
            tts_generator=tts_generator,
            elevenlabs_voice_id=elevenlabs_voice_id,
            bark_voice_id=bark_voice_id,
        )

    with gr.Row():
        save_config_btn = gr.Button("💾")
        save_config_btn.click(
            save_config,
            inputs=[conf.openai_model_state, conf.openai_temperature_state, conf.tts_generator_state,
                    conf.elevenlabs_voice_id_state, conf.bark_voice_id_state],
            outputs=None,
            queue=False
        )

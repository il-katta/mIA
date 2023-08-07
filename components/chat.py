import gradio as gr

import config
from bot import MiaBot


def is_available():
    return True


def gui(bot: 'MiaBot', conf: config.Config):

    chatbot = gr.Chatbot([], elem_id="chatbot")
    audio_out = gr.Audio(
        container=False,
        autoplay=True,
    )

    with gr.Row():
        with gr.Column(scale=5, min_width=100):
            message_textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
                lines=5,
            )
        with gr.Column(scale=1, min_width=20):
            submit_button = gr.Button("↗️", variant="primary")
            clear_button = gr.Button("🗑", variant="secondary")

        with gr.Column(scale=1, min_width=20):
            upload_btn = gr.UploadButton(
                "📁",
                file_types=[
                    # image", "video", "audio", 
                    "text"
                ]
            )
    message_textbox.submit(
        bot.on_message_pre,
        inputs=[chatbot, message_textbox],
        outputs=[chatbot, message_textbox],
        queue=False
    ).then(
        bot.on_message,
        inputs=[
            chatbot,
            conf.tts_generator_state,
            conf.elevenlabs_voice_id_state,
            conf.bark_voice_id_state
        ],
        outputs=[chatbot, audio_out]
    ).then(
        lambda: gr.update(interactive=True),
        inputs=None,
        outputs=[message_textbox],
        queue=False
    )
    submit_button.click(
        bot.on_message_pre,
        inputs=[chatbot, message_textbox],
        outputs=[chatbot, message_textbox],
        queue=False
    ).then(
        bot.on_message,
        inputs=[
            chatbot,
            conf.tts_generator_state,
            conf.elevenlabs_voice_id_state,
            conf.bark_voice_id_state
        ],
        outputs=[chatbot, audio_out]
    ).then(
        lambda: gr.update(interactive=True),
        inputs=None,
        outputs=[message_textbox],
        queue=False
    )

    upload_btn.upload(
        bot.on_file_pre,
        inputs=[chatbot, upload_btn],
        outputs=[chatbot],
        queue=False
    ).then(
        bot.on_message,
        inputs=[chatbot, conf.tts_generator_state],
        outputs=[chatbot, audio_out]
    )
    clear_button.click(
        lambda: ("", [],),
        inputs=[],
        outputs=[message_textbox, chatbot]
    )

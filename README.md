# mIA

## Features

* chatbot: openai chatbot mainly build with [langchain](https://www.langchain.com/) with 
  * text to speech functionality using [elevenlabs service](https://elevenlabs.io/) or [bark project](https://github.com/suno-ai/bark)
  * math chain to solve math problems
* image generation from song title and artist name - generate descriptive images of a song using openai GPT, stable diffusion, and prompt optimization language models
* image background removal using [rembg](https://github.com/danielgatis/rembg)
* image upscaler using `stabilityai/stable-diffusion-x4-upscaler` and `stabilityai/sd-x2-latent-upscaler` models
* read and write invisible watermarking of a image using [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) 


## Installation

* install python3.10 (https://www.python.org/downloads/)

* create virtual environment as usual

with bash:
```shell
python -m venv .venv
source .venv/bin/activate
```

or with powershell/cmd:

```shell
python -m venv .venv
.venv\Scripts\activate
```

* install requirements ( you can choose to install dependencies for any module of the project )
    for example install requirements for the base module only enable the chat module
```shell
pip install -r requirements-base.txt
```

## Usage
* create a `.env` file from `.env.example` and fill it with your credentials

Run with
```shell
./.venv/bin/python main.py
```

Open browser at http://localhost:1988

dev mode ( automatic reload on code change )
```shell
./.venv/bin/gradio main.py
```

## TODO
* riprisitinare la funzionalità dialogo con il bot ( ora il LLM non sembra avere visibilità dello storico della conversazione )
* valutare di aggiungere https://github.com/photosynthesis-team/piq alla generazione delle immagini per decidere quali scartare ?
* aggiungere il modulo per utilizzare roop https://github.com/s0md3v/roop
* aggiungere un esempio di interrogazione di documenti da LLM 
* migliorare e documentare le api rest create automaticamente da gradio 
* provare https://github.com/nagadomi/nunif per upscaling delle immagini
* image colorizer using https://github.com/richzhang/colorization ( or other opensource project ?? )

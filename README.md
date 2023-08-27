# mIA

[![Docker](https://github.com/il-katta/mIA/actions/workflows/docker-build.yml/badge.svg)](https://github.com/il-katta/mIA/actions/workflows/docker-build.yml)

## Features

* chatbot: openai chatbot mainly build with [langchain](https://www.langchain.com/) with 
  * text to speech functionality using [elevenlabs service](https://elevenlabs.io/) or [bark project](https://github.com/suno-ai/bark)
  * math chain to solve math problems
* image generation from text
* image generation from song title and artist name - generate descriptive images of a song using openai GPT, stable diffusion, and prompt optimization language models
* image background removal using [rembg](https://github.com/danielgatis/rembg)
* image upscaler using `stabilityai/stable-diffusion-x4-upscaler` and `stabilityai/sd-x2-latent-upscaler` models
* read and write invisible watermarking of an image using [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) 
* music generation using [audiocraft](https://github.com/facebookresearch/audiocraft) 

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

## Run with docker

### Run pre-built image

```shell
mkdir -p data
docker run \
  -p 1988:1988 \
  --env-file .env \
  --name mia \
  --user $(id -u):$(id -g) \
  -v "$(pwd)/data:/app/data" \
  --gpus all \
  ghcr.io/il-katta/mia:latest
```

### Build and run image

```shell
docker-compose up --pull always --build
```

## TODO
* [WIP] Zero-Shot Object Detection with Grounding DINO
* [WIP] aggiungere https://github.com/haoheliu/audioldm2
* [WIP] Image Tagger: fixare WaifuDiffusionTagger 
* https://github.com/IceClear/StableSR
* https://github.com/jantic/DeOldify
* riprisitinare la funzionalità dialogo con il bot ( ora il LLM non sembra avere visibilità dello storico della conversazione )
* valutare di aggiungere https://github.com/photosynthesis-team/piq alla generazione delle immagini per decidere quali scartare ?
* aggiungere il modulo per utilizzare roop https://github.com/s0md3v/roop
* aggiungere un esempio di interrogazione di documenti da LLM 
* migliorare e documentare le api rest create automaticamente da gradio 
* provare https://github.com/nagadomi/nunif per upscaling delle immagini
* image colorizer using https://github.com/richzhang/colorization ( or other opensource project ?? )
* viper gpt https://github.com/cvlab-columbia/viper
* fine tuning di facebook/audiogen-medium con dati da https://freesound.org
* aggiungere generazione di TTS lunghi con bark ( https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb  )
* valutare il progetto https://github.com/coqui-ai/TTS per TTS
* aggiungere pix2pix https://huggingface.co/timbrooks/instruct-pix2pix
* aggiungere https://huggingface.co/DeepFloyd/IF-I-XL-v1.0 con https://github.com/LuChengTHU/dpm-solver/blob/main/README.md
* aggiungere il supporto a https://modal.com/ 
* [audio-diffusion](https://github.com/teticio/audio-diffusion) ( [usage with pipelines](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audio_diffusion) ) 
* [dance diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/dance_diffusion)
* [kandinsky v2.2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky_v22)
* https://github.com/MushroomFleet/Deforum-Sequence-Tools
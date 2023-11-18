from typing import Optional, List

import whisper
import yt_dlp.postprocessor
from yt_dlp import YoutubeDL

import config
from utils._interfaces import DisposableModel
from utils._torch_utils import cuda_garbage_collection


class Video2Text(DisposableModel):
    _model = None
    _model_name = None

    def __init__(self):
        pass

    @cuda_garbage_collection
    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None

    def video2audio(self, url) -> str:
        class ProgressHook(yt_dlp.postprocessor.PostProcessor):
            title: Optional[str] = None
            filename: Optional[str] = None
            filepath: Optional[str] = None
            description: Optional[str] = None
            thumbnail: Optional[str] = None
            channel_id: Optional[str] = None
            duration: Optional[int] = None

            def __call__(self, d):
                info = d.get('info_dict')
                self.run(info)

            def run(self, info):
                self.title = info.get('title')
                if '_filename' in info:
                    self.filename = info.get('_filename')
                if 'filename' in info:
                    self.filename = info.get('filename')
                if 'filepath' in info:
                    self.filepath = info.get('filepath')
                self.description = info.get('description', None)
                self.thumbnail = info.get('thumbnail', None)
                self.channel_id = info.get('channel_id', None)
                self.duration = info.get('duration', None)

                return [], info

        pp = ProgressHook()

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(config.DATA_DIR / 'audio' / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            # "progress_hooks": [pp],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.add_post_processor(pp, when='post_process')
            result = ydl.download([url])
        if result != 0:
            raise RuntimeError("Failed to download audio")
        return pp.filepath

    def audio2text(self, audio_path, model_name="base"):
        if self._model_name != model_name:
            self.unload_model()
            self._model_name = model_name
        if self._model is None:
            self._model = whisper.load_model(
                model_name,
                download_root=config.DATA_DIR / "whisper",
            )
        #audio = whisper.load_audio(audio_path)
        #audio = whisper.pad_or_trim(audio)
        #text = self._model.transcribe(audio, verbose=True)
        text = self._model.transcribe(audio_path, verbose=True)
        return text["text"]

    def video2text(self, url, model_name="base"):
        audio_path = self.video2audio(url)
        text = self.audio2text(audio_path, model_name)
        return text

    @staticmethod
    def available_models() -> List[str]:
        return whisper.available_models()

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Video2Text()
        return cls._instance

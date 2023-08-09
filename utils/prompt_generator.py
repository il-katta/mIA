import gc
import logging

import torch
from transformers import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
from typing import Optional

from utils._interfaces import DisposableModel


class PromptGenerator(DisposableModel):
    pipe: Optional[TextGenerationPipeline] = None

    def __init__(self, model_name: str = "microsoft/Promptist"):
        """
        :param model_name: suggested model names is
            * "microsoft/Promptist"
            * "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator"
            * "Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator"
            * "Gustavosta/MagicPrompt-Stable-Diffusion"
            * "AUTOMATIC/promptgen-lexart"
            * "AUTOMATIC/promptgen-majinai-safe"
            * "AUTOMATIC/promptgen-majinai-unsafe"
        """
        self._logger = logging.getLogger(__name__)
        self.model_name = model_name

    def load_model(self):
        self._logger.info(f"Loading model {self.model_name}")
        self.pipe = pipeline("text-generation", model=self.model_name)

    def generate_prompt(self, incipit: str, max_length: int = 250) -> str:
        self._logger.info(f"Generating prompt for {self.model_name} with incipit {incipit}")
        if self.pipe is None:
            self.load_model()
        return self.pipe(incipit, max_length=max_length)[0]["generated_text"]

    def unload_model(self):
        self._logger.info(f"Unloading model {self.model_name}")
        if self.pipe:
            del self.pipe
            self.pipe = None
        self.gc()

    @staticmethod
    def gc():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def __del__(self):
        if self.pipe is not None:
            self.unload_model()

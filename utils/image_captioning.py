import logging
from pathlib import Path
from typing import Optional, Type, Union, Any, Tuple, List, Dict, Callable

import math
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, BlipProcessor, BlipForConditionalGeneration, ProcessorMixin, \
    PreTrainedModel, Blip2ForConditionalGeneration, Blip2Processor, AutoModelForCausalLM, AutoProcessor

import config
from utils._interfaces import DisposableModel
from utils._torch_utils import cuda_garbage_collection, torch_optimizer

__all__ = [
    "ImageCaptioning",
    "BLIPLargeCaptioning",
    "BLIP2Captioning",
    "GITLargeCaptioning",
    "WaifuDiffusionTagger",
    "DepDanbooruTagger"
]


class _Tagger(DisposableModel):
    model_path: Path

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = config.DATA_DIR / "huggingface"

    @property
    def name(self):
        return self.model_name

    @property
    def full_name(self):
        return f"{self.__class__.__name__}({self.name})"

    def predict(self, image: Image.Image) -> List[str]:
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def unload_model(self):
        raise NotImplementedError()

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        cuda_garbage_collection()

    def __str__(self):
        return self.full_name


class _Captioning(_Tagger):
    ProcessorClass: Type[Union[ProcessorMixin, AutoProcessor]]
    ModelClass: Type[PreTrainedModel]
    model_name: str
    processor: Optional[Union[Blip2Processor, BlipProcessor, Any]]
    model: Optional[PreTrainedModel]
    device: torch.device
    load_in_4bit: bool

    def __init__(
            self,
            processors_class: Type[Union[ProcessorMixin, AutoProcessor]],
            model_class: Type[PreTrainedModel],
            model_name: str,
            device: str = "cuda",
            load_in_4bit: bool = True
    ):
        super().__init__(model_name)
        self.ProcessorClass = processors_class
        self.ModelClass = model_class
        self.processor = None
        self.model = None
        self.device: torch.device = torch.device(device)
        self.load_in_4bit = load_in_4bit

    def load_model(self, model_name: Optional[str] = None):
        if model_name is None and self.model is not None and self.processor is not None:
            return
        if self.model_name == model_name and self.model is not None and self.processor is not None:
            return
        if model_name is not None:
            self.model_name = model_name
        if self.processor is not None or self.model is not None:
            self.unload_model()
        logging.info(f"Loading model {self}")
        self.processor = self.ProcessorClass.from_pretrained(
            self.model_name,
            cache_dir=self.model_path,
            **self.get_load_parameters()
        )
        self.model = self.ModelClass.from_pretrained(
            self.model_name,
            cache_dir=self.model_path,
            **self.get_load_parameters()
        )
        if not self.load_in_4bit:
            self.model = self.model.to(self.device)

    @cuda_garbage_collection
    def unload_model(self):
        if self.model is not None:
            logging.info(f"Unloading model {self}")
            del self.model
        self.model = None
        if self.processor is not None:
            del self.processor
        self.processor = None

    @torch_optimizer
    def predict(self, image: Image.Image, processor_arguments: Optional[Dict[str, Any]] = None):
        self.load_model()
        if self.model is None or self.processor is None:
            return ""
        with torch.no_grad(), torch.autocast(self.device.type):
            if processor_arguments is None:
                processor_arguments = {}
            inputs = self.processor(images=image, return_tensors="pt", **processor_arguments).to(self.device)
            ids = self.model.generate(**inputs)
            result = self.processor.batch_decode(ids, skip_special_tokens=True)
            logging.info(f"Predicted {result}")
            return result

    def get_load_parameters(self) -> Dict[str, Any]:
        if self.load_in_4bit:
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
                "device_map": "auto",
                "low_cpu_mem_usage": self.load_in_4bit
            }
        else:
            return {}


class BLIPLargeCaptioning(_Captioning):

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = "cuda",
                 load_in_4bit: bool = True):
        super().__init__(BlipProcessor, BlipForConditionalGeneration, model_name, device, load_in_4bit)

    def get_load_parameters(self) -> Dict[str, Any]:
        params = super().get_load_parameters()
        params["device_map"] = None
        return params


class BLIP2Captioning(_Captioning):
    def __init__(self, model_name: str, device: str = "cuda", load_in_4bit: bool = True):
        super().__init__(Blip2Processor, Blip2ForConditionalGeneration, model_name, device, load_in_4bit)

    def get_load_parameters(self) -> Dict[str, Any]:
        params = super().get_load_parameters()
        # TODO: fix this (??)
        if self.load_in_4bit:
            if self.device.type == "cuda":
                if self.device.index is not None:
                    params['device_map'] = f"cuda:{self.device.index}"
                else:
                    params['device_map'] = "cuda:0"
            else:
                params['device_map'] = self.device.type
        else:
            params['torch_dtype'] = torch.float16
        return params

    @torch_optimizer
    def predict(self, image: Image.Image, processor_arguments: Optional[Dict[str, Any]] = None):
        self.load_model()
        if self.model is None or self.processor is None:
            return ""
        with torch.no_grad(), torch.autocast(self.device.type):
            if processor_arguments is None:
                processor_arguments = {}
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                text="the image of", **processor_arguments
            ).to(self.device, torch.float16)
            output = self.model.generate(**inputs)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            logging.info(f"Predicted {result}")
            return result


class GITLargeCaptioning(_Captioning):

    def __init__(self, model_name: str = "microsoft/git-large-coco", device: str = "cuda", load_in_4bit: bool = True):
        super().__init__(AutoProcessor, AutoModelForCausalLM, model_name, device, load_in_4bit)

    def get_load_parameters(self) -> Dict[str, Any]:
        params = super().get_load_parameters()
        params["device_map"] = None
        return params


def resize_and_fill(image: Image.Image, size: Tuple[int, int]):
    width, height = size
    scale_w, scale_h = width / image.width, height / image.height
    resized_w, resized_h = width, height
    if scale_w < scale_h:
        resized_h = image.height * resized_w // image.width
    elif scale_h < scale_w:
        resized_w = image.width * resized_h // image.height
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)
    if resized_w == width and resized_h == height:
        return resized

    fill_l = math.floor((width - resized_w) / 2)
    fill_r = width - resized_w - fill_l
    fill_t = math.floor((height - resized_h) / 2)
    fill_b = height - resized_h - fill_t
    result = Image.new("RGB", (width, height))
    result.paste(resized, (fill_l, fill_t))
    if fill_t > 0:
        result.paste(resized.resize((width, fill_t), box=(0, 0, width, 0)), (0, 0))
    if fill_b > 0:
        result.paste(
            resized.resize(
                (width, fill_b), box=(0, resized.height, width, resized.height)
            ),
            (0, resized.height + fill_t),
        )
    if fill_l > 0:
        result.paste(resized.resize((fill_l, height), box=(0, 0, 0, height)), (0, 0))
    if fill_r > 0:
        result.paste(
            resized.resize(
                (fill_r, height), box=(resized.width, 0, resized.width, height)
            ),
            (resized.width + fill_l, 0),
        )
    return np.array(result, dtype=np.float32)


class WaifuDiffusionTagger(_Tagger):
    # from https://huggingface.co/spaces/deepghs/wd14_tagging_online/blob/main/app.py

    tags: Optional[pd.DataFrame] = None
    model: Optional[ort.InferenceSession] = None
    device: torch.device
    threshold: float

    def __init__(self, model_name: str, threshold: float, device: str = "cuda"):
        super().__init__(model_name)
        self.model = None
        self.device: torch.device = torch.device(device)
        self.threshold = threshold

    def load_model(self, model_name: Optional[str] = None):
        import huggingface_hub
        if not self.model or self.model_name != model_name:
            if model_name is not None:
                self.model_name = model_name
            path_model = huggingface_hub.hf_hub_download(
                self.model_name, filename="model.onnx", cache_dir=self.model_path
            )
            tags_path = huggingface_hub.hf_hub_download(
                self.model_name, filename="selected_tags.csv", cache_dir=self.model_path
            )
            if self.device.type == "cuda":
                providers = [
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]

            self.model = ort.InferenceSession(str(path_model), providers=providers)
            self.tags = pd.read_csv(tags_path)["name"].tolist()

    @cuda_garbage_collection
    def unload_model(self):
        if self.model is not None:
            del self.model
        self.model = None
        if self.tags is not None:
            del self.tags
        self.tags = None

    def predict(self, image: Image.Image):
        self.load_model()
        if self.model is None or self.tags is None:
            return ""

        _, height, width, _ = self.model.get_inputs()[0].shape

        image_np = resize_and_fill(image.convert("RGB"), (width, height))
        image_np = image_np[:, :, ::-1]
        image_np = np.expand_dims(image_np, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image_np})[0]
        labels: list[Tuple[str, float]] = list(zip(self.tags, probs[0].astype(float)))

        return [lab[0] for lab in labels if lab[1] >= self.threshold]


# https://github.com/AUTOMATIC1111/TorchDeepDanbooru


class DeepDanbooruModel(nn.Module):
    def __init__(self):
        super(DeepDanbooruModel, self).__init__()

        self.tags = []

        self.n_Conv_0 = nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2))
        self.n_MaxPool_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.n_Conv_1 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_2 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=64)
        self.n_Conv_3 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_4 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_5 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_6 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_7 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_8 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_9 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_10 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_11 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=512, stride=(2, 2))
        self.n_Conv_12 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=128)
        self.n_Conv_13 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(2, 2))
        self.n_Conv_14 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_15 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_16 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_17 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_18 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_19 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_20 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_21 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_22 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_23 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_24 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_25 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_26 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_27 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_28 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_29 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_30 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_31 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_33 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_34 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_35 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_36 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=1024, stride=(2, 2))
        self.n_Conv_37 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=256)
        self.n_Conv_38 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_39 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_40 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_41 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_42 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_43 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_44 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_45 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_46 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_47 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_48 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_49 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_50 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_51 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_52 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_53 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_54 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_55 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_56 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_57 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_58 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_59 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_60 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_61 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_62 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_63 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_65 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_66 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_67 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_68 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_69 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_70 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_71 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_72 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_73 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_74 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_75 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_76 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_77 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_78 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_79 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_80 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_81 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_82 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_83 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_84 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_85 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_86 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_87 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_88 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_89 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_90 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_91 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_92 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_93 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_94 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_95 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_96 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_97 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_98 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_99 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_100 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_101 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_102 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_103 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_104 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_105 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_106 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_107 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_108 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_109 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_110 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_111 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_112 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_113 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_114 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_115 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_116 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_117 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_118 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_119 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_120 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_121 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_122 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_123 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_124 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_125 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_126 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_127 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_129 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_130 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_131 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_132 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_133 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_134 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_135 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_136 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_137 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_138 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_139 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_140 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_141 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_142 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_143 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_144 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_145 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_146 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_147 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_148 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_149 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_150 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_151 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_152 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_153 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_154 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_155 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_156 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_157 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_158 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=2048, stride=(2, 2))
        self.n_Conv_159 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=512)
        self.n_Conv_160 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(2, 2))
        self.n_Conv_161 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_162 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_163 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_164 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_165 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_166 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_167 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_168 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=4096, stride=(2, 2))
        self.n_Conv_169 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=1024)
        self.n_Conv_170 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_171 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_172 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_173 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_174 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_175 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_176 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_177 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=9176, bias=False)

    def forward(self, *inputs):
        t_358, = inputs
        t_359 = t_358.permute(*[0, 3, 1, 2])
        t_359_padded = F.pad(t_359, [2, 3, 2, 3], value=0)
        t_360 = self.n_Conv_0(t_359_padded)
        t_361 = F.relu(t_360)
        t_361 = F.pad(t_361, [0, 1, 0, 1], value=float('-inf'))
        t_362 = self.n_MaxPool_0(t_361)
        t_363 = self.n_Conv_1(t_362)
        t_364 = self.n_Conv_2(t_362)
        t_365 = F.relu(t_364)
        t_365_padded = F.pad(t_365, [1, 1, 1, 1], value=0)
        t_366 = self.n_Conv_3(t_365_padded)
        t_367 = F.relu(t_366)
        t_368 = self.n_Conv_4(t_367)
        t_369 = torch.add(t_368, t_363)
        t_370 = F.relu(t_369)
        t_371 = self.n_Conv_5(t_370)
        t_372 = F.relu(t_371)
        t_372_padded = F.pad(t_372, [1, 1, 1, 1], value=0)
        t_373 = self.n_Conv_6(t_372_padded)
        t_374 = F.relu(t_373)
        t_375 = self.n_Conv_7(t_374)
        t_376 = torch.add(t_375, t_370)
        t_377 = F.relu(t_376)
        t_378 = self.n_Conv_8(t_377)
        t_379 = F.relu(t_378)
        t_379_padded = F.pad(t_379, [1, 1, 1, 1], value=0)
        t_380 = self.n_Conv_9(t_379_padded)
        t_381 = F.relu(t_380)
        t_382 = self.n_Conv_10(t_381)
        t_383 = torch.add(t_382, t_377)
        t_384 = F.relu(t_383)
        t_385 = self.n_Conv_11(t_384)
        t_386 = self.n_Conv_12(t_384)
        t_387 = F.relu(t_386)
        t_387_padded = F.pad(t_387, [0, 1, 0, 1], value=0)
        t_388 = self.n_Conv_13(t_387_padded)
        t_389 = F.relu(t_388)
        t_390 = self.n_Conv_14(t_389)
        t_391 = torch.add(t_390, t_385)
        t_392 = F.relu(t_391)
        t_393 = self.n_Conv_15(t_392)
        t_394 = F.relu(t_393)
        t_394_padded = F.pad(t_394, [1, 1, 1, 1], value=0)
        t_395 = self.n_Conv_16(t_394_padded)
        t_396 = F.relu(t_395)
        t_397 = self.n_Conv_17(t_396)
        t_398 = torch.add(t_397, t_392)
        t_399 = F.relu(t_398)
        t_400 = self.n_Conv_18(t_399)
        t_401 = F.relu(t_400)
        t_401_padded = F.pad(t_401, [1, 1, 1, 1], value=0)
        t_402 = self.n_Conv_19(t_401_padded)
        t_403 = F.relu(t_402)
        t_404 = self.n_Conv_20(t_403)
        t_405 = torch.add(t_404, t_399)
        t_406 = F.relu(t_405)
        t_407 = self.n_Conv_21(t_406)
        t_408 = F.relu(t_407)
        t_408_padded = F.pad(t_408, [1, 1, 1, 1], value=0)
        t_409 = self.n_Conv_22(t_408_padded)
        t_410 = F.relu(t_409)
        t_411 = self.n_Conv_23(t_410)
        t_412 = torch.add(t_411, t_406)
        t_413 = F.relu(t_412)
        t_414 = self.n_Conv_24(t_413)
        t_415 = F.relu(t_414)
        t_415_padded = F.pad(t_415, [1, 1, 1, 1], value=0)
        t_416 = self.n_Conv_25(t_415_padded)
        t_417 = F.relu(t_416)
        t_418 = self.n_Conv_26(t_417)
        t_419 = torch.add(t_418, t_413)
        t_420 = F.relu(t_419)
        t_421 = self.n_Conv_27(t_420)
        t_422 = F.relu(t_421)
        t_422_padded = F.pad(t_422, [1, 1, 1, 1], value=0)
        t_423 = self.n_Conv_28(t_422_padded)
        t_424 = F.relu(t_423)
        t_425 = self.n_Conv_29(t_424)
        t_426 = torch.add(t_425, t_420)
        t_427 = F.relu(t_426)
        t_428 = self.n_Conv_30(t_427)
        t_429 = F.relu(t_428)
        t_429_padded = F.pad(t_429, [1, 1, 1, 1], value=0)
        t_430 = self.n_Conv_31(t_429_padded)
        t_431 = F.relu(t_430)
        t_432 = self.n_Conv_32(t_431)
        t_433 = torch.add(t_432, t_427)
        t_434 = F.relu(t_433)
        t_435 = self.n_Conv_33(t_434)
        t_436 = F.relu(t_435)
        t_436_padded = F.pad(t_436, [1, 1, 1, 1], value=0)
        t_437 = self.n_Conv_34(t_436_padded)
        t_438 = F.relu(t_437)
        t_439 = self.n_Conv_35(t_438)
        t_440 = torch.add(t_439, t_434)
        t_441 = F.relu(t_440)
        t_442 = self.n_Conv_36(t_441)
        t_443 = self.n_Conv_37(t_441)
        t_444 = F.relu(t_443)
        t_444_padded = F.pad(t_444, [0, 1, 0, 1], value=0)
        t_445 = self.n_Conv_38(t_444_padded)
        t_446 = F.relu(t_445)
        t_447 = self.n_Conv_39(t_446)
        t_448 = torch.add(t_447, t_442)
        t_449 = F.relu(t_448)
        t_450 = self.n_Conv_40(t_449)
        t_451 = F.relu(t_450)
        t_451_padded = F.pad(t_451, [1, 1, 1, 1], value=0)
        t_452 = self.n_Conv_41(t_451_padded)
        t_453 = F.relu(t_452)
        t_454 = self.n_Conv_42(t_453)
        t_455 = torch.add(t_454, t_449)
        t_456 = F.relu(t_455)
        t_457 = self.n_Conv_43(t_456)
        t_458 = F.relu(t_457)
        t_458_padded = F.pad(t_458, [1, 1, 1, 1], value=0)
        t_459 = self.n_Conv_44(t_458_padded)
        t_460 = F.relu(t_459)
        t_461 = self.n_Conv_45(t_460)
        t_462 = torch.add(t_461, t_456)
        t_463 = F.relu(t_462)
        t_464 = self.n_Conv_46(t_463)
        t_465 = F.relu(t_464)
        t_465_padded = F.pad(t_465, [1, 1, 1, 1], value=0)
        t_466 = self.n_Conv_47(t_465_padded)
        t_467 = F.relu(t_466)
        t_468 = self.n_Conv_48(t_467)
        t_469 = torch.add(t_468, t_463)
        t_470 = F.relu(t_469)
        t_471 = self.n_Conv_49(t_470)
        t_472 = F.relu(t_471)
        t_472_padded = F.pad(t_472, [1, 1, 1, 1], value=0)
        t_473 = self.n_Conv_50(t_472_padded)
        t_474 = F.relu(t_473)
        t_475 = self.n_Conv_51(t_474)
        t_476 = torch.add(t_475, t_470)
        t_477 = F.relu(t_476)
        t_478 = self.n_Conv_52(t_477)
        t_479 = F.relu(t_478)
        t_479_padded = F.pad(t_479, [1, 1, 1, 1], value=0)
        t_480 = self.n_Conv_53(t_479_padded)
        t_481 = F.relu(t_480)
        t_482 = self.n_Conv_54(t_481)
        t_483 = torch.add(t_482, t_477)
        t_484 = F.relu(t_483)
        t_485 = self.n_Conv_55(t_484)
        t_486 = F.relu(t_485)
        t_486_padded = F.pad(t_486, [1, 1, 1, 1], value=0)
        t_487 = self.n_Conv_56(t_486_padded)
        t_488 = F.relu(t_487)
        t_489 = self.n_Conv_57(t_488)
        t_490 = torch.add(t_489, t_484)
        t_491 = F.relu(t_490)
        t_492 = self.n_Conv_58(t_491)
        t_493 = F.relu(t_492)
        t_493_padded = F.pad(t_493, [1, 1, 1, 1], value=0)
        t_494 = self.n_Conv_59(t_493_padded)
        t_495 = F.relu(t_494)
        t_496 = self.n_Conv_60(t_495)
        t_497 = torch.add(t_496, t_491)
        t_498 = F.relu(t_497)
        t_499 = self.n_Conv_61(t_498)
        t_500 = F.relu(t_499)
        t_500_padded = F.pad(t_500, [1, 1, 1, 1], value=0)
        t_501 = self.n_Conv_62(t_500_padded)
        t_502 = F.relu(t_501)
        t_503 = self.n_Conv_63(t_502)
        t_504 = torch.add(t_503, t_498)
        t_505 = F.relu(t_504)
        t_506 = self.n_Conv_64(t_505)
        t_507 = F.relu(t_506)
        t_507_padded = F.pad(t_507, [1, 1, 1, 1], value=0)
        t_508 = self.n_Conv_65(t_507_padded)
        t_509 = F.relu(t_508)
        t_510 = self.n_Conv_66(t_509)
        t_511 = torch.add(t_510, t_505)
        t_512 = F.relu(t_511)
        t_513 = self.n_Conv_67(t_512)
        t_514 = F.relu(t_513)
        t_514_padded = F.pad(t_514, [1, 1, 1, 1], value=0)
        t_515 = self.n_Conv_68(t_514_padded)
        t_516 = F.relu(t_515)
        t_517 = self.n_Conv_69(t_516)
        t_518 = torch.add(t_517, t_512)
        t_519 = F.relu(t_518)
        t_520 = self.n_Conv_70(t_519)
        t_521 = F.relu(t_520)
        t_521_padded = F.pad(t_521, [1, 1, 1, 1], value=0)
        t_522 = self.n_Conv_71(t_521_padded)
        t_523 = F.relu(t_522)
        t_524 = self.n_Conv_72(t_523)
        t_525 = torch.add(t_524, t_519)
        t_526 = F.relu(t_525)
        t_527 = self.n_Conv_73(t_526)
        t_528 = F.relu(t_527)
        t_528_padded = F.pad(t_528, [1, 1, 1, 1], value=0)
        t_529 = self.n_Conv_74(t_528_padded)
        t_530 = F.relu(t_529)
        t_531 = self.n_Conv_75(t_530)
        t_532 = torch.add(t_531, t_526)
        t_533 = F.relu(t_532)
        t_534 = self.n_Conv_76(t_533)
        t_535 = F.relu(t_534)
        t_535_padded = F.pad(t_535, [1, 1, 1, 1], value=0)
        t_536 = self.n_Conv_77(t_535_padded)
        t_537 = F.relu(t_536)
        t_538 = self.n_Conv_78(t_537)
        t_539 = torch.add(t_538, t_533)
        t_540 = F.relu(t_539)
        t_541 = self.n_Conv_79(t_540)
        t_542 = F.relu(t_541)
        t_542_padded = F.pad(t_542, [1, 1, 1, 1], value=0)
        t_543 = self.n_Conv_80(t_542_padded)
        t_544 = F.relu(t_543)
        t_545 = self.n_Conv_81(t_544)
        t_546 = torch.add(t_545, t_540)
        t_547 = F.relu(t_546)
        t_548 = self.n_Conv_82(t_547)
        t_549 = F.relu(t_548)
        t_549_padded = F.pad(t_549, [1, 1, 1, 1], value=0)
        t_550 = self.n_Conv_83(t_549_padded)
        t_551 = F.relu(t_550)
        t_552 = self.n_Conv_84(t_551)
        t_553 = torch.add(t_552, t_547)
        t_554 = F.relu(t_553)
        t_555 = self.n_Conv_85(t_554)
        t_556 = F.relu(t_555)
        t_556_padded = F.pad(t_556, [1, 1, 1, 1], value=0)
        t_557 = self.n_Conv_86(t_556_padded)
        t_558 = F.relu(t_557)
        t_559 = self.n_Conv_87(t_558)
        t_560 = torch.add(t_559, t_554)
        t_561 = F.relu(t_560)
        t_562 = self.n_Conv_88(t_561)
        t_563 = F.relu(t_562)
        t_563_padded = F.pad(t_563, [1, 1, 1, 1], value=0)
        t_564 = self.n_Conv_89(t_563_padded)
        t_565 = F.relu(t_564)
        t_566 = self.n_Conv_90(t_565)
        t_567 = torch.add(t_566, t_561)
        t_568 = F.relu(t_567)
        t_569 = self.n_Conv_91(t_568)
        t_570 = F.relu(t_569)
        t_570_padded = F.pad(t_570, [1, 1, 1, 1], value=0)
        t_571 = self.n_Conv_92(t_570_padded)
        t_572 = F.relu(t_571)
        t_573 = self.n_Conv_93(t_572)
        t_574 = torch.add(t_573, t_568)
        t_575 = F.relu(t_574)
        t_576 = self.n_Conv_94(t_575)
        t_577 = F.relu(t_576)
        t_577_padded = F.pad(t_577, [1, 1, 1, 1], value=0)
        t_578 = self.n_Conv_95(t_577_padded)
        t_579 = F.relu(t_578)
        t_580 = self.n_Conv_96(t_579)
        t_581 = torch.add(t_580, t_575)
        t_582 = F.relu(t_581)
        t_583 = self.n_Conv_97(t_582)
        t_584 = F.relu(t_583)
        t_584_padded = F.pad(t_584, [0, 1, 0, 1], value=0)
        t_585 = self.n_Conv_98(t_584_padded)
        t_586 = F.relu(t_585)
        t_587 = self.n_Conv_99(t_586)
        t_588 = self.n_Conv_100(t_582)
        t_589 = torch.add(t_587, t_588)
        t_590 = F.relu(t_589)
        t_591 = self.n_Conv_101(t_590)
        t_592 = F.relu(t_591)
        t_592_padded = F.pad(t_592, [1, 1, 1, 1], value=0)
        t_593 = self.n_Conv_102(t_592_padded)
        t_594 = F.relu(t_593)
        t_595 = self.n_Conv_103(t_594)
        t_596 = torch.add(t_595, t_590)
        t_597 = F.relu(t_596)
        t_598 = self.n_Conv_104(t_597)
        t_599 = F.relu(t_598)
        t_599_padded = F.pad(t_599, [1, 1, 1, 1], value=0)
        t_600 = self.n_Conv_105(t_599_padded)
        t_601 = F.relu(t_600)
        t_602 = self.n_Conv_106(t_601)
        t_603 = torch.add(t_602, t_597)
        t_604 = F.relu(t_603)
        t_605 = self.n_Conv_107(t_604)
        t_606 = F.relu(t_605)
        t_606_padded = F.pad(t_606, [1, 1, 1, 1], value=0)
        t_607 = self.n_Conv_108(t_606_padded)
        t_608 = F.relu(t_607)
        t_609 = self.n_Conv_109(t_608)
        t_610 = torch.add(t_609, t_604)
        t_611 = F.relu(t_610)
        t_612 = self.n_Conv_110(t_611)
        t_613 = F.relu(t_612)
        t_613_padded = F.pad(t_613, [1, 1, 1, 1], value=0)
        t_614 = self.n_Conv_111(t_613_padded)
        t_615 = F.relu(t_614)
        t_616 = self.n_Conv_112(t_615)
        t_617 = torch.add(t_616, t_611)
        t_618 = F.relu(t_617)
        t_619 = self.n_Conv_113(t_618)
        t_620 = F.relu(t_619)
        t_620_padded = F.pad(t_620, [1, 1, 1, 1], value=0)
        t_621 = self.n_Conv_114(t_620_padded)
        t_622 = F.relu(t_621)
        t_623 = self.n_Conv_115(t_622)
        t_624 = torch.add(t_623, t_618)
        t_625 = F.relu(t_624)
        t_626 = self.n_Conv_116(t_625)
        t_627 = F.relu(t_626)
        t_627_padded = F.pad(t_627, [1, 1, 1, 1], value=0)
        t_628 = self.n_Conv_117(t_627_padded)
        t_629 = F.relu(t_628)
        t_630 = self.n_Conv_118(t_629)
        t_631 = torch.add(t_630, t_625)
        t_632 = F.relu(t_631)
        t_633 = self.n_Conv_119(t_632)
        t_634 = F.relu(t_633)
        t_634_padded = F.pad(t_634, [1, 1, 1, 1], value=0)
        t_635 = self.n_Conv_120(t_634_padded)
        t_636 = F.relu(t_635)
        t_637 = self.n_Conv_121(t_636)
        t_638 = torch.add(t_637, t_632)
        t_639 = F.relu(t_638)
        t_640 = self.n_Conv_122(t_639)
        t_641 = F.relu(t_640)
        t_641_padded = F.pad(t_641, [1, 1, 1, 1], value=0)
        t_642 = self.n_Conv_123(t_641_padded)
        t_643 = F.relu(t_642)
        t_644 = self.n_Conv_124(t_643)
        t_645 = torch.add(t_644, t_639)
        t_646 = F.relu(t_645)
        t_647 = self.n_Conv_125(t_646)
        t_648 = F.relu(t_647)
        t_648_padded = F.pad(t_648, [1, 1, 1, 1], value=0)
        t_649 = self.n_Conv_126(t_648_padded)
        t_650 = F.relu(t_649)
        t_651 = self.n_Conv_127(t_650)
        t_652 = torch.add(t_651, t_646)
        t_653 = F.relu(t_652)
        t_654 = self.n_Conv_128(t_653)
        t_655 = F.relu(t_654)
        t_655_padded = F.pad(t_655, [1, 1, 1, 1], value=0)
        t_656 = self.n_Conv_129(t_655_padded)
        t_657 = F.relu(t_656)
        t_658 = self.n_Conv_130(t_657)
        t_659 = torch.add(t_658, t_653)
        t_660 = F.relu(t_659)
        t_661 = self.n_Conv_131(t_660)
        t_662 = F.relu(t_661)
        t_662_padded = F.pad(t_662, [1, 1, 1, 1], value=0)
        t_663 = self.n_Conv_132(t_662_padded)
        t_664 = F.relu(t_663)
        t_665 = self.n_Conv_133(t_664)
        t_666 = torch.add(t_665, t_660)
        t_667 = F.relu(t_666)
        t_668 = self.n_Conv_134(t_667)
        t_669 = F.relu(t_668)
        t_669_padded = F.pad(t_669, [1, 1, 1, 1], value=0)
        t_670 = self.n_Conv_135(t_669_padded)
        t_671 = F.relu(t_670)
        t_672 = self.n_Conv_136(t_671)
        t_673 = torch.add(t_672, t_667)
        t_674 = F.relu(t_673)
        t_675 = self.n_Conv_137(t_674)
        t_676 = F.relu(t_675)
        t_676_padded = F.pad(t_676, [1, 1, 1, 1], value=0)
        t_677 = self.n_Conv_138(t_676_padded)
        t_678 = F.relu(t_677)
        t_679 = self.n_Conv_139(t_678)
        t_680 = torch.add(t_679, t_674)
        t_681 = F.relu(t_680)
        t_682 = self.n_Conv_140(t_681)
        t_683 = F.relu(t_682)
        t_683_padded = F.pad(t_683, [1, 1, 1, 1], value=0)
        t_684 = self.n_Conv_141(t_683_padded)
        t_685 = F.relu(t_684)
        t_686 = self.n_Conv_142(t_685)
        t_687 = torch.add(t_686, t_681)
        t_688 = F.relu(t_687)
        t_689 = self.n_Conv_143(t_688)
        t_690 = F.relu(t_689)
        t_690_padded = F.pad(t_690, [1, 1, 1, 1], value=0)
        t_691 = self.n_Conv_144(t_690_padded)
        t_692 = F.relu(t_691)
        t_693 = self.n_Conv_145(t_692)
        t_694 = torch.add(t_693, t_688)
        t_695 = F.relu(t_694)
        t_696 = self.n_Conv_146(t_695)
        t_697 = F.relu(t_696)
        t_697_padded = F.pad(t_697, [1, 1, 1, 1], value=0)
        t_698 = self.n_Conv_147(t_697_padded)
        t_699 = F.relu(t_698)
        t_700 = self.n_Conv_148(t_699)
        t_701 = torch.add(t_700, t_695)
        t_702 = F.relu(t_701)
        t_703 = self.n_Conv_149(t_702)
        t_704 = F.relu(t_703)
        t_704_padded = F.pad(t_704, [1, 1, 1, 1], value=0)
        t_705 = self.n_Conv_150(t_704_padded)
        t_706 = F.relu(t_705)
        t_707 = self.n_Conv_151(t_706)
        t_708 = torch.add(t_707, t_702)
        t_709 = F.relu(t_708)
        t_710 = self.n_Conv_152(t_709)
        t_711 = F.relu(t_710)
        t_711_padded = F.pad(t_711, [1, 1, 1, 1], value=0)
        t_712 = self.n_Conv_153(t_711_padded)
        t_713 = F.relu(t_712)
        t_714 = self.n_Conv_154(t_713)
        t_715 = torch.add(t_714, t_709)
        t_716 = F.relu(t_715)
        t_717 = self.n_Conv_155(t_716)
        t_718 = F.relu(t_717)
        t_718_padded = F.pad(t_718, [1, 1, 1, 1], value=0)
        t_719 = self.n_Conv_156(t_718_padded)
        t_720 = F.relu(t_719)
        t_721 = self.n_Conv_157(t_720)
        t_722 = torch.add(t_721, t_716)
        t_723 = F.relu(t_722)
        t_724 = self.n_Conv_158(t_723)
        t_725 = self.n_Conv_159(t_723)
        t_726 = F.relu(t_725)
        t_726_padded = F.pad(t_726, [0, 1, 0, 1], value=0)
        t_727 = self.n_Conv_160(t_726_padded)
        t_728 = F.relu(t_727)
        t_729 = self.n_Conv_161(t_728)
        t_730 = torch.add(t_729, t_724)
        t_731 = F.relu(t_730)
        t_732 = self.n_Conv_162(t_731)
        t_733 = F.relu(t_732)
        t_733_padded = F.pad(t_733, [1, 1, 1, 1], value=0)
        t_734 = self.n_Conv_163(t_733_padded)
        t_735 = F.relu(t_734)
        t_736 = self.n_Conv_164(t_735)
        t_737 = torch.add(t_736, t_731)
        t_738 = F.relu(t_737)
        t_739 = self.n_Conv_165(t_738)
        t_740 = F.relu(t_739)
        t_740_padded = F.pad(t_740, [1, 1, 1, 1], value=0)
        t_741 = self.n_Conv_166(t_740_padded)
        t_742 = F.relu(t_741)
        t_743 = self.n_Conv_167(t_742)
        t_744 = torch.add(t_743, t_738)
        t_745 = F.relu(t_744)
        t_746 = self.n_Conv_168(t_745)
        t_747 = self.n_Conv_169(t_745)
        t_748 = F.relu(t_747)
        t_748_padded = F.pad(t_748, [0, 1, 0, 1], value=0)
        t_749 = self.n_Conv_170(t_748_padded)
        t_750 = F.relu(t_749)
        t_751 = self.n_Conv_171(t_750)
        t_752 = torch.add(t_751, t_746)
        t_753 = F.relu(t_752)
        t_754 = self.n_Conv_172(t_753)
        t_755 = F.relu(t_754)
        t_755_padded = F.pad(t_755, [1, 1, 1, 1], value=0)
        t_756 = self.n_Conv_173(t_755_padded)
        t_757 = F.relu(t_756)
        t_758 = self.n_Conv_174(t_757)
        t_759 = torch.add(t_758, t_753)
        t_760 = F.relu(t_759)
        t_761 = self.n_Conv_175(t_760)
        t_762 = F.relu(t_761)
        t_762_padded = F.pad(t_762, [1, 1, 1, 1], value=0)
        t_763 = self.n_Conv_176(t_762_padded)
        t_764 = F.relu(t_763)
        t_765 = self.n_Conv_177(t_764)
        t_766 = torch.add(t_765, t_760)
        t_767 = F.relu(t_766)
        t_768 = self.n_Conv_178(t_767)
        t_769 = F.avg_pool2d(t_768, kernel_size=t_768.shape[-2:])
        t_770 = torch.squeeze(t_769, 3)
        t_770 = torch.squeeze(t_770, 2)
        t_771 = torch.sigmoid(t_770)
        return t_771

    def load_state_dict(self, state_dict, **kwargs):
        self.tags = state_dict.get('tags', [])

        super(DeepDanbooruModel, self).load_state_dict({k: v for k, v in state_dict.items() if k != 'tags'})


# TODO: use the multi gpu version from https://huggingface.co/skytnt/deepdanbooru_onnx

class DepDanbooruTagger(_Tagger):
    threshold: float
    model: Optional[DeepDanbooruModel]
    device: torch.device

    def __init__(self, threshold: float, device: str = "cuda"):
        super().__init__("deepdanbooru")
        self.model = None
        self.device = torch.device(device)
        self.threshold = threshold

    def _load(self, model_path: Union[Path, str], model_url: str, progress: bool = True, force_download: bool = False):
        model_path = Path(model_path)
        if model_path.exists():
            return model_path

        if model_url is not None and (force_download or not model_path.is_file()):
            if not model_path.parent.is_dir():
                model_path.parent.mkdir(parents=True)
            torch.hub.download_url_to_file(model_url, str(model_path), progress=progress)
            return model_path

        return model_path

    def load_model(self):
        file = self._load(
            model_path=self.model_path / "model-resnet_custom_v3.pt",
            model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt'
        )
        self.model = DeepDanbooruModel()
        self.model.load_state_dict(torch.load(file, map_location='cpu'))
        self.model.eval()
        self.model.to(self.device, torch.float16)

    @cuda_garbage_collection
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

    @torch_optimizer
    def predict(self, image: Image.Image):
        self.load_model()
        image = resize_and_fill(image, (512, 512))
        image_np = np.expand_dims(np.array(image, dtype=np.float32), 0) / 255
        x = torch.from_numpy(image_np).half().to(self.device)
        y = self.model(x)[0].detach().cpu().numpy()
        labels = list(zip(self.model.tags, y))
        return [label[0] for label in labels if label[1] >= self.threshold]


class ImageCaptioning(DisposableModel):
    models: List[_Tagger] = []
    _load_in_4bit: bool
    _device: str

    banned_tags = ["unused"]

    tag_translations = {
        "rating:safe": "sfw",
    }

    def __init__(self, load_in_4bit: bool = True, device: str = "cuda"):
        self._load_in_4bit = load_in_4bit
        self._device = device
        self.init_models(load_in_4bit=load_in_4bit)

    def init_models(self, load_in_4bit: bool = None, device: str = None):
        if load_in_4bit == self._load_in_4bit and self._device == device and len(self.models) > 0:
            return
        if load_in_4bit is not None:
            self._load_in_4bit = load_in_4bit
        if device is not None:
            self._device = device
        models = [BLIPLargeCaptioning(load_in_4bit=self._load_in_4bit, device=self._device)]

        for model_name in [
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-2.7b-coco",
            "Salesforce/blip2-opt-6.7b",
            "Salesforce/blip2-opt-6.7b-coco",
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip2-flan-t5-xl-coco",
            "Salesforce/blip2-flan-t5-xxl",
        ]:
            models.append(BLIP2Captioning(model_name, load_in_4bit=self._load_in_4bit, device=self._device))

        models.append(GITLargeCaptioning(load_in_4bit=self._load_in_4bit, device=self._device))
        models.append(DepDanbooruTagger(0.5, device=self._device))

        for model_name, threshold in [
            ("SmilingWolf/wd-v1-4-vit-tagger", 0.35),
            ("SmilingWolf/wd-v1-4-convnext-tagger", 0.35),
            ("SmilingWolf/wd-v1-4-vit-tagger-v2", 0.3537),
            ("SmilingWolf/wd-v1-4-convnext-tagger-v2", 0.3685),
            ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", 0.3771),
        ]:
            models.append(WaifuDiffusionTagger(model_name, threshold, device=self._device))
        self.unload_models()
        self.models = models

    @cuda_garbage_collection
    def unload_model(self):
        for model in self.models:
            model.unload_model()

    def unload_models(self):
        self.unload_model()

    def load_models(self, models_choose: List[str]):
        for model in self.models:
            if model.full_name in models_choose:
                try:
                    model.load_model()
                except Exception as e:
                    logging.error(f"Failed to load model {model.name}")
                    logging.exception(e)
                finally:
                    model.unload_model()

    @property
    def model_names(self) -> List[str]:
        return [model.full_name for model in self.models]

    def predict_image(
            self,
            image,
            models_choose: List[str],
            callback_after_model_unload: Optional[Callable[[_Tagger], None]] = None
    ) -> List[str]:
        predictions = set()
        for model in self.models:
            if model.full_name in models_choose:
                # with model:
                try:
                    model.load_model()
                    for tag in self.filter_tags(model.predict(image)):
                        predictions.add(tag)
                finally:
                    model.unload_model()
                    if callback_after_model_unload is not None:
                        callback_after_model_unload(model)

        return list(predictions)

    def tag_directory(
            self,
            input_directory_path: Union[str, Path],
            output_directory_path: str,
            models_choose: List[str],
            resize: bool = True,
            resize_size: int = 512,
            overwrite: bool = False,
            keep_existing_tags: bool = True,
            additional_tags_prefix: str = "",
            additional_tags_postfix: str = "",
            callback_after_model_unload: Optional[Callable[[_Tagger], None]] = None
    ):
        input_directory_path = Path(input_directory_path)
        if not input_directory_path.exists():
            return "[Directory does not exist]"
        if not input_directory_path.is_dir():
            return "[Path is not a directory]"

        logging.info("copy all the images to target directory")
        output_directory_path = Path(output_directory_path)
        if not output_directory_path.exists():
            output_directory_path.mkdir(parents=True, exist_ok=True)
        images: List[Tuple[str, Image.Image]] = []
        for path in input_directory_path.iterdir():
            if self.is_image(path):
                images.append((
                    path.name,
                    self.copy_image_to_directory(path, output_directory_path, resize, resize_size, overwrite)
                ))

        predictions: Dict[str, set] = {}
        for model in tqdm([model for model in self.models if model.full_name in models_choose], "models"):
            # with model:
            try:
                model.load_model()
                for name, image in tqdm(images, f"tagging images with {model}"):
                    try:
                        if name not in predictions.keys():
                            predictions[name] = set()
                        for tag in self.filter_tags(model.predict(image)):
                            predictions[name].add(tag)
                    except Exception as e:
                        logging.error(f"Failed to predict {name}")
                        logging.exception(e)
            finally:
                model.unload_model()
                if callback_after_model_unload is not None:
                    callback_after_model_unload(model)

        def _str_to_list(string: str) -> List[str]:
            if len(string.strip()) > 0:
                return [t.strip() for t in string.split(",") if t.strip() != ""]
            else:
                return []

        def _join_sets(*sets: Union[set, List]) -> List:
            result = set()
            for s in sets:
                result = result.union(s)
            return list(result)

        additional_tags_prefix = _str_to_list(additional_tags_prefix)
        additional_tags_postfix = _str_to_list(additional_tags_postfix)

        for filename, tags in tqdm(predictions.items(), "writing tags"):
            tag_file = output_directory_path / Path(filename).with_suffix(".txt")
            if tag_file.exists() and keep_existing_tags:
                additional_tags_prefix += _str_to_list(tag_file.read_text("utf-8").strip())
            final_tags = _join_sets(additional_tags_prefix, tags, additional_tags_postfix)
            tag_file.write_text(", ".join(final_tags), "utf-8")

    @staticmethod
    def clean_tag(tag: Optional[str]) -> str | None:
        if not tag:
            return None
        return str(tag).rstrip("\n").rstrip(".").rstrip(",").rstrip(" ").replace("_", " ")

    @classmethod
    def filter_tags(cls, predict: List[str] | Dict | str) -> List[str]:
        if isinstance(predict, str):
            tags = [predict]
        elif isinstance(predict, dict):
            tags = list(predict.keys())
        elif isinstance(predict, list):
            tags = predict
        else:
            return []

        # return [clean_tag(tag) for tag in tags if clean_tag(tag) != "" and clean_tag(tag) != None]
        for tag in tags:
            tag = cls.clean_tag(tag)
            if tag == "" or tag is None:
                continue
            for banned in cls.banned_tags:
                if banned in tag:
                    continue
            if tag in cls.tag_translations.keys():
                tag = cls.tag_translations[tag]
            yield tag

    @staticmethod
    def resize_image(image: Image.Image, size: int) -> Image.Image:
        if image.width > image.height:
            return image.resize((size, int(image.height * (size / image.width))), Image.Resampling.LANCZOS)
        else:
            return image.resize((int(image.width * (size / image.height)), size), Image.Resampling.LANCZOS)

    @staticmethod
    def is_image(filepath: Path) -> bool:
        if not filepath.is_file():
            return False
        return filepath.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]

    @classmethod
    def copy_image_to_directory(cls, image_path: Path, target_dir: Path, resize: bool = True, resize_size: int = 512,
                                overwrite: bool = False) -> Image.Image:
        target_path = target_dir / image_path.name
        if not target_path.exists() or overwrite:
            image = Image.open(image_path).convert("RGB")
            if resize:
                image = cls.resize_image(image, resize_size)
                image.save(target_path)
        else:
            image = Image.open(target_path).convert("RGB")
        return image

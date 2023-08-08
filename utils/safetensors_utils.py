from typing import Dict, Any, Union
import mmap
import torch
import json

DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool
}


class SafetensorsUtils(object):

    @staticmethod
    def create_tensor(storage, info, offset):
        """Creates a tensor without holding on to an open handle to the parent model
        file."""
        dtype = DTYPES[info["dtype"]]
        shape = info["shape"]
        start, stop = info["data_offsets"]
        return (
            torch.asarray(storage[start + offset: stop + offset], dtype=torch.uint8)
            .view(dtype=dtype)
            .reshape(shape)
            .clone()
            .detach()
        )

    @staticmethod
    def read_metadata(filename: str) -> Union[Dict[str, Any], None]:
        with open(filename, mode="r", encoding="utf8") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                header = m.read(8)
                n = int.from_bytes(header, "little")
                metadata_bytes = m.read(n)
                metadata = json.loads(metadata_bytes)
        md = metadata.get("__metadata__", {})
        return md

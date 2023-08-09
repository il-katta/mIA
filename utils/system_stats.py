import logging
from typing import Optional, Tuple, List
import gc

from utils import package_exists
from utils._interfaces import DisposableModel

if package_exists("torch"):
    import torch
else:
    torch = None

if package_exists("nvidia_smi"):
    import nvidia_smi


else:
    nvidia_smi = None


class SystemStats(object):
    _gpu_handle = None
    _gpu_id: int = None
    _torch_device: Optional[torch.cuda.device] = None

    def __init__(self, gpu_id=0):
        if nvidia_smi:
            nvidia_smi.nvmlInit()
        self.change_gpu(gpu_id)

    def _init_torch(self):
        if torch and torch.cuda.is_available():
            torch.cuda.init()
            self._torch_device = torch.cuda.device(f"cuda")
            self._torch_device.__enter__()

    def _exit_torch(self):
        if self._torch_device:
            self._torch_device.__exit__(None, None, None)
            del self._torch_device
            self._torch_device = None

    def change_gpu(self, gpu_id):
        if nvidia_smi:
            self._gpu_handle = self._get_gpu_handle(gpu_id)
        if self._gpu_id != gpu_id:
            self._gpu_id = gpu_id
            self._exit_torch()
            self._init_torch()

    def _get_gpu_handle(self, gpu_id: Optional[int] = None):
        if gpu_id is None or (gpu_id == self._gpu_id and self._gpu_handle is not None):
            return self._gpu_handle
        else:
            return nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

    def get_gpu_name(self, gpu_id: Optional[int] = None) -> Optional[str]:
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetName(self._get_gpu_handle(gpu_id)).decode('utf-8')
        if torch:
            return torch.cuda.get_device_name(self._gpu_id if gpu_id is None else gpu_id)
        return None

    def get_gpu_temperature(self, gpu_id: Optional[int] = None) -> Optional[int]:
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetTemperature(self._get_gpu_handle(gpu_id), 0)
        return None

    def get_gpu_fan_speed(self, gpu_id: Optional[int] = None) -> Optional[int]:
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetFanSpeed(self._get_gpu_handle(gpu_id))
        return None

    def get_power_usage(self, gpu_id: Optional[int] = None) -> Tuple[int, int]:
        if nvidia_smi:
            actual = int(nvidia_smi.nvmlDeviceGetPowerUsage(self._get_gpu_handle(gpu_id)) / 1000)
            total = int(nvidia_smi.nvmlDeviceGetPowerManagementLimit(self._get_gpu_handle(gpu_id)) / 1000)
            return actual, total
        return 0, 0

    def get_gpu_ram_usage(self, gpu_id: Optional[int] = None) -> Tuple[int, int]:
        if nvidia_smi:
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(self._get_gpu_handle(gpu_id))
            allocated = mem_info.used
            total = mem_info.total
        elif torch:
            allocated = torch.cuda.memory_allocated(self._gpu_id if gpu_id is None else gpu_id)
            total = torch.cuda.get_device_properties(self._gpu_id if gpu_id is None else gpu_id).total_memory
        else:
            allocated = 0
            total = 0

        return allocated, total

    def free_vram(self):
        for model in self._disposable_models:
            try:
                model.unload_model()
            except Exception as ex:
                logging.exception(ex)

        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def get_gpu_count() -> int:
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetCount()
        if torch:
            return torch.cuda.device_count()
        return 0

    _disposable_models: List[DisposableModel] = []

    def register_disposable_model(self, model: DisposableModel):
        self._disposable_models.append(model)

import logging
from typing import Optional, Tuple, List, Union
import gc

import pandas as pd

from utils import package_exists, cuda_is_available
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
    _history: pd.DataFrame = pd.DataFrame
    _enabled = False

    def __init__(self, gpu_id=0, store_history=False):
        self._store_history = store_history
        self._enabled = cuda_is_available()
        if self._enabled:
            nvidia_smi.nvmlInit()
        self.change_gpu(gpu_id)
        if self._store_history:
            # create new Dataframe with columns for each metric and gpu_id and timestamp as index
            self._history = pd.DataFrame(
                columns=["gpu_id", "timestamp", "gpu_temperature", "gpu_fan_speed", "power_usage_actual",
                         "power_usage_total", "gpu_ram_usage_allocated", "gpu_ram_usage_total"])
            self._history.set_index(["timestamp", "gpu_id"], inplace=True)

    def _add_to_history(self, gpu_id: Optional[int] = None,
                        gpu_temperature: Optional[int] = None,
                        gpu_fan_speed: Optional[int] = None,
                        power_usage_actual: Optional[int] = None,
                        power_usage_total: Optional[int] = None,
                        gpu_ram_usage_allocated: Optional[int] = None,
                        gpu_ram_usage_total: Optional[int] = None):
        if self._store_history:
            # append new row to history
            gpu_id = gpu_id if gpu_id is not None else self._gpu_id
            new_row = {
                "gpu_id": gpu_id,
                "timestamp": pd.Timestamp.now(),
                "gpu_temperature": gpu_temperature,
                "gpu_fan_speed": gpu_fan_speed,
                "power_usage_actual": power_usage_actual,
                "power_usage_total": power_usage_total,
                "gpu_ram_usage_allocated": gpu_ram_usage_allocated,
                "gpu_ram_usage_total": gpu_ram_usage_total
            }

            self._history = pd.concat(
                [
                    self._history,
                    pd.DataFrame([new_row], index=["timestamp", "gpu_id"])
                ],
                ignore_index=True
            )

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
        if not self._enabled:
            return
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
        if not self._enabled:
            return None
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetName(self._get_gpu_handle(gpu_id)).decode('utf-8')
        if torch:
            return torch.cuda.get_device_name(self._gpu_id if gpu_id is None else gpu_id)
        return None

    def get_gpu_temperature(self, gpu_id: Optional[int] = None) -> Optional[int]:
        if not self._enabled:
            return None
        if nvidia_smi:
            value = nvidia_smi.nvmlDeviceGetTemperature(self._get_gpu_handle(gpu_id), 0)
            self._add_to_history(gpu_id, gpu_temperature=value)
            return value
        return None

    def get_gpu_temperature_history(self, gpu_id: Optional[int] = None) -> pd.DataFrame:
        return self._get_history(["gpu_temperature"], gpu_id)

    def get_gpu_fan_speed(self, gpu_id: Optional[int] = None) -> Optional[int]:
        if not self._enabled:
            return None
        if nvidia_smi:
            value = nvidia_smi.nvmlDeviceGetFanSpeed(self._get_gpu_handle(gpu_id))
            self._add_to_history(gpu_id, gpu_fan_speed=value)
            return value
        return None

    def get_gpu_fan_speed_history(self, gpu_id: Optional[int] = None) -> pd.DataFrame:
        return self._get_history(["gpu_fan_speed"], gpu_id)

    def get_power_usage(self, gpu_id: Optional[int] = None) -> Tuple[int, int]:
        if not self._enabled:
            return 0, 0
        if nvidia_smi:
            actual = int(nvidia_smi.nvmlDeviceGetPowerUsage(self._get_gpu_handle(gpu_id)) / 1000)
            total = int(nvidia_smi.nvmlDeviceGetPowerManagementLimit(self._get_gpu_handle(gpu_id)) / 1000)
            self._add_to_history(gpu_id, power_usage_actual=actual, power_usage_total=total)
            return actual, total
        return 0, 0

    def get_power_usage_history(self, gpu_id: Optional[int] = None) -> pd.DataFrame:
        return self._get_history(["power_usage_actual", "power_usage_total"], gpu_id)

    def get_gpu_ram_usage(self, gpu_id: Optional[int] = None) -> Tuple[int, int]:
        if not self._enabled:
            return 0, 0
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
        if total > 0:
            self._add_to_history(gpu_id, gpu_ram_usage_allocated=allocated, gpu_ram_usage_total=total)
        return allocated, total

    def get_gpu_ram_usage_history(self, gpu_id: Optional[int] = None) -> pd.DataFrame:
        return self._get_history(["gpu_ram_usage_allocated", "gpu_ram_usage_total"], gpu_id)

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

    def get_processes(self, gpu_id: Optional[int] = None):
        if not self._enabled or not nvidia_smi:
            return []
        processes = nvidia_smi.nvmlDeviceGetGraphicsRunningProcesses(self._get_gpu_handle(gpu_id))
        processes.sort(
            key=lambda p: p.usedGpuMemory or 0,
            reverse=True
        )

        def get_process_name(pid: int) -> str:
            import pynvml
            try:
                return nvidia_smi.nvmlSystemGetProcessName(pid).decode('utf-8')
            except pynvml.NVMLError_NotFound:  # in docker container the above function fails
                return ""

        return [
            {
                "pid": p.pid,
                "process_name": get_process_name(p.pid),
                "used_gpu_memory": p.usedGpuMemory,
            }
            for p in processes
        ]

    def get_gpu_count(self) -> int:
        if not self._enabled:
            return 0
        if nvidia_smi:
            return nvidia_smi.nvmlDeviceGetCount()
        if torch:
            return torch.cuda.device_count()
        return 0

    _disposable_models: List[DisposableModel] = []

    def register_disposable_model(self, model: DisposableModel):
        self._disposable_models.append(model)

    def _get_history(self, fields: List[str], gpu_id: Optional[int] = None) -> pd.DataFrame:
        # max allowed in gradio is 5000
        # reduces to 1000 for performance and graphical reasons
        gpu_id = gpu_id if gpu_id is not None else self._gpu_id
        return self._history[self._history["gpu_id"] == gpu_id][fields + ["timestamp", "gpu_id"]].dropna().tail(1000)

import logging

import torch

import gc
import os

from app_config import AppConfig, CONFIG_FILE_NAME

LOGGER = logging.getLogger(__name__)


def _reload_cuda_visible_devices():
    """Reload cuda_visible_devices from config before any GPU operation."""
    try:
        AppConfig(CONFIG_FILE_NAME).reload_cuda_visible_devices()
    except Exception:
        pass


def get_gpu_device():
    _reload_cuda_visible_devices()
    if torch.cuda.is_available():
        gpu_device = 'cuda'
    elif torch.backends.mps.is_available():
        gpu_device = 'mps'
    else:
        gpu_device = 'cpu'
    return gpu_device


def pick_device_with_info() -> str:
    _reload_cuda_visible_devices()
    # Apple Silicon (Mac)
    if torch.backends.mps.is_available():
        LOGGER.info("MPS is available. Using Apple's GPU (mps).")
        LOGGER.info("PyTorch version: %s", torch.__version__)
        LOGGER.info("MPS backend: available (no CUDA-style GPU ids on macOS/MPS)")
        return "mps"

    # NVIDIA CUDA
    if torch.cuda.is_available():
        LOGGER.info("MPS not found. Using CUDA (cuda).")
        LOGGER.info("PyTorch version: %s", torch.__version__)
        LOGGER.info("CUDA compiled version: %s", torch.version.cuda)
        LOGGER.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

        num_gpus = torch.cuda.device_count()
        LOGGER.info("Number of visible CUDA GPUs: %d", num_gpus)
        LOGGER.info("Visible GPU ids: %s", list(range(num_gpus)))

        for gpu_id in range(num_gpus):
            props = torch.cuda.get_device_properties(gpu_id)
            total_gb = props.total_memory / (1024 ** 3)
            try:
                free_b, total_b = torch.cuda.mem_get_info(gpu_id)
                used_gb = (total_b - free_b) / (1024 ** 3)
                total_gb_actual = total_b / (1024 ** 3)
                LOGGER.info(
                    "GPU %d: %s | CC %d.%d | VRAM %.2f GB | used %.2f GB / %.2f GB (%.1f%%)",
                    gpu_id, props.name, props.major, props.minor, total_gb,
                    used_gb, total_gb_actual, 100 * used_gb / total_gb_actual if total_gb_actual > 0 else 0
                )
            except Exception:
                LOGGER.info(
                    "GPU %d: %s | CC %d.%d | VRAM %.2f GB",
                    gpu_id, props.name, props.major, props.minor, total_gb
                )

        return "cuda"

    # CPU fallback
    LOGGER.info("MPS and CUDA not found. Using CPU.")
    LOGGER.info("PyTorch version: %s", torch.__version__)
    return "cpu"

def clean_memory_with_info():
    LOGGER.info("MEMORY PRIOR CLEANING")
    pick_device_with_info()

    LOGGER.info("MEMORY CLEANING")
    # Clear Python garbage
    gc.collect()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("MEMORY AFTER CLEANING")
    pick_device_with_info()


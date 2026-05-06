import os
import logging
import numpy as np
import torch
import faiss
#from .utils import l2_normalize

LOGGER = logging.getLogger(__name__)


def ensure_2d_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N,D). Got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def log_gpu_info() -> None:
    LOGGER.info("PyTorch version: %s", torch.__version__)
    LOGGER.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    if torch.cuda.is_available():
        LOGGER.info("CUDA available. torch.version.cuda=%s", torch.version.cuda)
        n = torch.cuda.device_count()
        LOGGER.info("Visible CUDA GPUs: %d", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            LOGGER.info("GPU %d: %s | CC %d.%d | VRAM %.2f GB",
                        i, props.name, props.major, props.minor, props.total_memory / (1024 ** 3))
        cur = torch.cuda.current_device()
        LOGGER.info("Current CUDA device: %d (%s)", cur, torch.cuda.get_device_name(cur))
        try:
            free_b, total_b = torch.cuda.mem_get_info(cur)
            LOGGER.info("CUDA mem: free %.2f GB / total %.2f GB",
                        free_b / (1024 ** 3), total_b / (1024 ** 3))
        except Exception:
            LOGGER.debug("torch.cuda.mem_get_info not available", exc_info=True)
    else:
        LOGGER.info("CUDA not available in torch.")

    try:
        import faiss  # type: ignore
        LOGGER.info("FAISS version: %s", getattr(faiss, "__version__", "<unknown>"))
        try:
            LOGGER.info("FAISS detected GPUs: %s", faiss.get_num_gpus())
        except Exception:
            LOGGER.info("FAISS get_num_gpus failed (maybe CPU-only build).")
    except Exception:
        LOGGER.info("FAISS not importable at GPU log stage.")


def _build_faiss_index_ip(X: np.ndarray, use_gpu: bool = True):
    """
    Build FAISS index for inner-product search (cosine if X is L2-normalized).
    Returns (index, use_gpu_actual).
    """


    d = X.shape[1]
    index = faiss.IndexFlatIP(d)

    use_gpu_actual = False
    if use_gpu:
        try:
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                use_gpu_actual = True
        except Exception:
            use_gpu_actual = False

    index.add(X)
    return index, use_gpu_actual


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True):
    """Build FAISS index for cosine similarity search by L2-normalizing embeddings and using inner-product index.
    """
    log_gpu_info()
    #X = l2_normalize(embeddings)
    # This is faster than our numpy implementation, and modifies in-place.
    # Note that FAISS's normalize_L2 does not handle NaNs or Infs, so we assume the input is clean.
    # Very important is that the normelized embeddings can now be used with the IndexFlatIP to get cosine similarity search.
    faiss.normalize_L2(embeddings)
    index, use_gpu_actual = _build_faiss_index_ip(embeddings, use_gpu=use_gpu and not torch.backends.mps.is_available())
    LOGGER.info("Built FAISS IndexFlatIP (cosine via L2+IP). gpu=%s", use_gpu_actual)
    return index

import torch
from .log import logger


def select_device(min_memory=2047, experimental=False):
    logger_instance = logger.get_logger()

    if torch.cuda.is_available():
        selected_gpu = None
        max_free_memory = -1

        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.mem_get_info(i)[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                selected_gpu = i

        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger_instance.warning(
                f"GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left. Switching to CPU."
            )
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{selected_gpu}")

    if torch.backends.mps.is_available():
        if experimental:
            logger_instance.warn("experimental: found Apple GPU, using MPS.")
            return torch.device("mps")
        else:
            logger_instance.info("found Apple GPU, but using CPU.")
            return torch.device("cpu")

    logger_instance.warning("no GPU found, using CPU instead")
    return torch.device("cpu")

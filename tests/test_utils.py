import pytest
import torch


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: requires {gpu_count}"
    local_gpu_count: int = torch.cuda.device_count()
    return pytest.mark.skipif(local_gpu_count < gpu_count, reason=message)

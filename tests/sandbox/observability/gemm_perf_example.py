"""Example: measure GEMM performance with Tracer.

Based on the observability README "Track Performance: Timing and Memory".
"""

import asyncio

import torch

from forge.observability import get_or_create_metric_logger
from forge.observability.perf_tracker import Tracer


def _pick_device() -> tuple[str, str, torch.dtype, bool]:
    if torch.xpu.is_available():
        return "xpu", "gpu", torch.float16, True
    return "cpu", "cpu", torch.float32, False


async def main() -> None:
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one({"console": {"logging_mode": "global_reduce"}})

    device, timer, dtype, track_memory = _pick_device()
    print(f"Using device: {device}, dtype: {dtype}, track_memory: {track_memory}")

    size = 2048
    iters = 10

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    tracer = Tracer(
        prefix=f"gemm_perf/{device}",
        track_memory=track_memory,
        timer=timer,
    )

    tracer.start()
    for _ in range(iters):
        _ = torch.mm(a, b)
        tracer.step("gemm")
    tracer.stop()

    await mlogger.flush.call_one(global_step=0)
    await mlogger.shutdown.call_one()


if __name__ == "__main__":
    asyncio.run(main())
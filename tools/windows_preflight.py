"""Check the Windows CUDA environment. No model is loaded or trained."""
from __future__ import annotations

import json
import platform
import sys


def require_windows(system: str | None = None) -> None:
    if (system or platform.system()) != "Windows":
        raise RuntimeError("Use the Windows RTX 5080 PC. Training on the Mac is not authorised.")


def check_cuda() -> dict:
    require_windows()
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and the locked Windows CUDA "
            "PyTorch build. Do not fall back to CPU training."
        )
    device = torch.device("cuda:0")
    # Exercise real kernels; is_available alone does not prove architecture support.
    with torch.inference_mode():
        a = torch.ones((32, 32), device=device)
        assert torch.allclose(a @ a, torch.full_like(a, 32))
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([.9, .8], device=device)
        assert torchvision.ops.nms(boxes, scores, .5).tolist() == [0]
        x = torch.ones((1, 1, 8, 8), device=device)
        kernel = torch.ones((1, 1, 3, 3), device=device)
        assert torch.allclose(torch.nn.functional.conv2d(x, kernel), torch.full((1, 1, 6, 6), 9., device=device))
        torch.cuda.synchronize()
    return {
        "status": "passed", "python": platform.python_version(), "os": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "checks": ["CUDA matrix multiplication", "CUDA torchvision NMS", "CUDA convolution"],
        "training_performed": False,
    }


def main() -> None:
    try:
        print(json.dumps(check_cuda(), indent=2))
    except Exception as exc:
        print(f"GPU preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

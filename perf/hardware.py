"""What this machine is: versions, device, environment, and a short probe of whether the allocator setting in force changes anything."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys

import torch

ENV_PREFIXES = ("PYTORCH_", "SYCL", "UR_", "ZE_", "ONEAPI", "ONEDNN", "IGC_", "NEO_", "XPU", "OMP_", "CCL_", "TORCH")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class HardwareReport:
    def collect(self) -> dict:
        info: dict = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                      "packages": {p: _version(p) for p in ("xpu-kernels", "intel-extension-for-pytorch", "safetensors", "numpy", "pillow", "transformers", "diffusers")},
                      "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}, "devices": []}
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                props = xpu.get_device_properties(index)
                info["devices"].append({"index": index, "name": props.name, **{k: getattr(props, k) for k in ("total_memory", "driver_version", "max_compute_units", "gpu_eu_count", "has_fp16", "has_fp64", "has_atomic64")
                                                                             if hasattr(props, k)}})
            info["xpu_graph"] = hasattr(xpu, "XPUGraph")
            info["xpu_int_mm"] = self._int_mm()
        info["cpu_count"] = os.cpu_count()
        return info

    @staticmethod
    def _int_mm() -> str:
        try:
            a = torch.randint(-5, 5, (32, 32), dtype=torch.int8, device="xpu")
            torch._int_mm(a, a)
            return "works"
        except Exception as error:  # noqa: BLE001 - the answer is the message
            return f"fails: {type(error).__name__}: {str(error).splitlines()[0][:100] if str(error) else ''}"

    def markdown(self, info: dict) -> str:
        out = ["# Hardware and software", "", f"- python {info['python']}, {info['platform']}, {info['cpu_count']} CPU threads", f"- torch {info['torch']}"]
        out += [f"- {name} {version or 'not installed'}" for name, version in info["packages"].items()]
        for d in info["devices"]:
            out.append(f"- device {d['index']}: {d['name']}, {d.get('total_memory', 0) / 2**20:.0f} MB, driver {d.get('driver_version', '?')}, "
                       f"{d.get('gpu_eu_count', d.get('max_compute_units', '?'))} compute units")
        if "xpu_graph" in info:
            out.append(f"- XPUGraph available: {info['xpu_graph']}; torch._int_mm: {info['xpu_int_mm']}")
        if not info["devices"]:
            out.append("- no XPU device visible to torch")
        out += ["", "Environment:", ""] + [f"- `{k}={v}`" for k, v in info["environment"].items()]
        return "\n".join(out) + "\n"


class AllocatorProbe:
    """Fragments the device allocator (alternating blocks freed, then larger blocks requested) and reports what it reserved. Run it once
    with ``PYTORCH_ALLOC_CONF=expandable_segments:True`` and once with ``:False`` exported: equal numbers mean the setting changes nothing here."""

    BLOCK_MB = 64
    BLOCKS = 24

    def __init__(self, device: str = "xpu", quick: bool = False):
        self.device, self.quick = device, quick

    def run(self) -> dict:
        module = getattr(torch, self.device.split(":")[0], None)
        block = (1 if self.quick else self.BLOCK_MB) * 2**20
        count = 6 if self.quick else self.BLOCKS
        stats = lambda: module.memory_stats() if module is not None and hasattr(module, "memory_stats") else {}  # noqa: E731

        def reserved() -> float:
            if module is not None and hasattr(module, "memory_reserved"):
                return module.memory_reserved() / 2**20
            return 0.0

        result = {"device": self.device, "alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_XPU_ALLOC_CONF") or "unset",
                  "block_mb": block / 2**20, "blocks": count}
        blocks = [torch.empty(block, dtype=torch.uint8, device=self.device) for _ in range(count)]
        result["reserved_full_mb"] = reserved()
        keep = blocks[::2]
        del blocks
        result["reserved_after_free_mb"] = reserved()
        big = [torch.empty(block * 2, dtype=torch.uint8, device=self.device) for _ in range(count // 2 - 1)]
        result["reserved_after_big_mb"] = reserved()
        result["retries"] = stats().get("num_alloc_retries", 0)
        del big, keep
        return result


def main_hardware() -> dict:
    return HardwareReport().collect()


if __name__ == "__main__":
    print(json.dumps(main_hardware(), indent=1))

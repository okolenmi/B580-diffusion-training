#!/usr/bin/env python3
"""Minimal A2: the smallest Kohya config that can produce a LoRA at all.

Context. The full comparison config (batch 2, 512x512, --full_bf16)
completed exactly ONE step on a 12,216 MB B580 and then stalled with the
allocator pinned at 12,191/12,216 MB -- one core at 100%, no GPU
progress. That is allocation thrashing, not slowness.

The memory that run held was dominated by things none of which are
needed to answer the question:

  * 264 text-encoder LoRA modules, trained. `--network_train_unet_only`
    drops them. This is not just a saving -- it also makes the arms
    MATCH, because this project's default LoRA adapts UNet attention only
    (93.6M params, 3.6% of the model, no text encoder, no conv/norm/FF).
    Kohya's default trains both towers, so the previous config was not a
    like-for-like comparison in the first place.
  * both CLIP towers kept resident for text-encoder training.
  * latent caching for every image at once.

So this is the minimal config, and every reduction is recorded with its
reason rather than applied silently.

Peak is sampled from the card while the run is live. A single reading
taken after setup is NOT the peak -- that mistake produced a wrong
"it needs 3.5 GB more" estimate earlier -- so sampling happens during the
training loop, repeatedly, and the maximum is reported.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
SD = Path("/home/okolenmi/Desktop/kohya/kohya_ss/sd-scripts")
PY = "/home/okolenmi/comfy/venv/bin/python"
CKPT = "/home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors"


class PeakSampler(threading.Thread):
    """Poll the card while the run is live; keep the maximum.

    Stopped by the caller when the subprocess exits. Daemon so a hung
    sampler cannot keep the script alive.
    """

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self.peak_used_mb = 0
        self.peak_reserved_mb = 0
        self.total_mb = 0
        self.samples = 0
        self._stop = threading.Event()

    def _read(self):
        import torch
        free, total = torch.xpu.mem_get_info()
        used = (total - free) // 1024 // 1024
        try:
            reserved = torch.xpu.max_memory_reserved() // 1024 // 1024
        except Exception:
            reserved = 0
        return used, reserved, total // 1024 // 1024

    def run(self):
        while not self._stop.is_set():
            try:
                used, reserved, total = self._read()
            except Exception:
                # The card can be busy enough that a query fails; that is
                # not a reason to abort the measurement.
                self._stop.wait(self.period)
                continue
            self.samples += 1
            self.total_mb = total
            self.peak_used_mb = max(self.peak_used_mb, used)
            self.peak_reserved_mb = max(self.peak_reserved_mb, reserved)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out_dir = REPO / "runs" / "a2_kohya" / "minimal"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "console.log"

    argv = [
        PY, "sdxl_train_network.py",
        f"--pretrained_model_name_or_path={CKPT}",
        f"--train_data_dir={REPO / 'runs' / 'a2_kohya' / 'images'}",
        f"--in_json={REPO / 'runs' / 'a2_kohya' / 'metadata.jsonl'}",
        f"--output_dir={out_dir}",
        "--output_name=minimal",
        # rank/alpha kept at Test_02's values so the resulting file is
        # comparable to the LoRA under investigation.
        "--network_dim=64", "--network_alpha=32",
        # UNet only: drops 264 TE modules AND matches this project's
        # attention-only UNet default. The biggest single saving here.
        "--network_train_unet_only",
        # smallest shapes that still produce a real 64x64 latent
        "--resolution=384,384", "--enable_bucket", "--bucket_no_upscale",
        "--min_timestep=1", "--max_timestep=999",
        "--train_batch_size=1",
        "--max_train_epochs=1", f"--max_train_steps={steps}",
        "--learning_rate=1e-4", "--optimizer_type=adamw",
        "--mixed_precision=bf16", "--full_bf16",
        # cache latents one at a time instead of all at once
        "--vae_batch_size=1",
        # fewer loader workers: each one is a separate process holding
        # decoded images, and they are what turns a tight fit into a
        # thrash. Also avoids the forkserver processes that outlived
        # earlier killed runs.
        "--max_data_loader_n_workers=1",
        "--seed=1234",
    ]

    sampler = PeakSampler()
    print(f"config: batch=1 res=384 unet_only full_bf16 steps={steps}", flush=True)
    t0 = time.monotonic()
    sampler.start()
    with log.open("w") as fh:
        proc = subprocess.run(argv, cwd=SD, stdout=fh, stderr=subprocess.STDOUT)
    sampler.stop()
    sampler.join(timeout=2)
    elapsed = time.monotonic() - t0

    text = log.read_text(errors="replace")
    outcome = "ok" if proc.returncode == 0 else "failed"
    for code, name in ((40, "OUT_OF_RESOURCES"), (20, "DEVICE_LOST")):
        if f"error: {code}" in text:
            outcome = f"level_zero_{name}"
    steps_done = 0
    m = re.findall(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", text)
    if m:
        steps_done = int(m[-1][0])

    result = {
        "config": {"batch": 1, "resolution": "384,384", "unet_only": True,
                   "full_bf16": True, "steps_requested": steps},
        "returncode": proc.returncode, "outcome": outcome,
        "steps_completed": steps_done, "seconds": round(elapsed, 1),
        "peak_card_used_mb": sampler.peak_used_mb,
        "peak_reserved_mb": sampler.peak_reserved_mb,
        "card_total_mb": sampler.total_mb,
        "samples": sampler.samples,
        "saved": sorted(p.name for p in out_dir.glob("*.safetensors")),
        "log": str(log),
    }
    (out_dir / "minimal.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if outcome == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
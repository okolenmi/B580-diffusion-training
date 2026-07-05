"""CLI entry point — simplified, config-driven."""

import argparse
import os
import random
import sys
from pathlib import Path

import torch

#os.environ["SYCL_CACHE_PERSISTENT"] = "1"
#os.environ["SYCL_CACHE_DIR"] = str(Path.home() / ".cache" / "sycl_kernels")
os.environ["SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD"] = "0"
os.environ["SYCL_CACHE_IN_MEM"] = "1"

os.environ["UR_L0_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["IGC_EnableDPEmulation"] = "1"
torch.set_num_threads(6)

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main():
    p = argparse.ArgumentParser(
        description="Distillation converter for SDXL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert.py --config my_run.toml
  python convert.py --config my_run.toml --lr 5e-4 --steps 5000
        """)

    p.add_argument("--config", type=str, default=None,
                   help="Path to TOML config file. Created with defaults if missing.")
    p.add_argument("--reset-config", type=str, default=None,
                   help="Overwrite config with current defaults and exit.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override total steps from config.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate from config.")
    p.add_argument("--mode", type=str,
                   choices=["distillation", "cyclic", "lora", "full"],
                   help="Override training mode.")
    p.add_argument("--fresh", action="store_true",
                   help="Force fresh start (ignore resume).")
    p.add_argument("--student", type=str,
                   help="Override student checkpoint path.")
    p.add_argument("--run-id", type=int, default=None,
                   help="Run ID for progress tracking.")
    p.add_argument("--start-from", type=str, default=None,
                   choices=["teacher", "student", "resume"],
                   help="Set start_from for this run (controls resume behavior "
                        "in Trainer -- see config_model.py's start_from docstring).")
    p.add_argument("--resume-optimizer", type=str, default=None,
                   help="Override resume optimizer path.")
    p.add_argument("--reset-optimizer", action="store_true", default=False,
                   help="Discard saved optimizer second-moment states.")

    cli = p.parse_args()

    if cli.reset_config:
        from .config_io import write_default_config
        write_default_config(cli.reset_config)
        sys.exit(0)

    if not cli.config:
        p.error("--config is required")

    config_path = Path(cli.config)

    if not config_path.exists():
        from .config_io import write_default_config
        write_default_config(config_path)
        print(f"  Edit {config_path} and run again.")
        sys.exit(0)

    from .config_io import read_config, write_config
    from .config_model import TrainingConfig

    config = read_config(config_path)

    # Apply CLI overrides
    changed = False
    if cli.mode:
        from .config_model import LoRATuning, CyclicTuning, DistillationTuning, FullTuning
        modes = {
            "lora": LoRATuning,
            "cyclic": CyclicTuning,
            "distillation": DistillationTuning,
            "full": FullTuning
        }
        if cli.mode in modes:
            # If the method is already the same, we might want to keep the existing settings,
            # but if it's different, we definitely need a new object.
            if config.tuning.method != cli.mode:
                config.tuning = modes[cli.mode]()
                changed = True

    if cli.steps is not None:
        config.common.steps = cli.steps
        changed = True
    if cli.lr is not None:
        config.common.lr = cli.lr
        changed = True
    if cli.student:
        config.paths.student = cli.student
        changed = True
    if cli.fresh:
        if not cli.student:
            config.paths.student = None
        config.paths.resume_checkpoint = None
        config.paths.resume_optimizer = None
        config.common.resume_step = 0
        config.start_from = "teacher"
        changed = True
    if cli.start_from:
        config.start_from = cli.start_from
        changed = True

    if cli.run_id is not None:
        # Store run_id in a private attribute that won't be saved to TOML
        # but can be accessed by the trainer.
        object.__setattr__(config.common, "_cli_run_id", cli.run_id)

    if cli.resume_optimizer:
        config.paths.resume_optimizer = cli.resume_optimizer
        changed = True
    if cli.reset_optimizer:
        config.paths.resume_optimizer = None
        changed = True

    if changed:
        write_config(config_path, config)

    random.seed(config.common.seed)
    torch.manual_seed(config.common.seed)

    # ComfyUI only needs to be on sys.path from here on (Trainer/UNet/CLIP/VAE
    # code lazily does `import comfy.xxx`). Deferring this to the last
    # possible moment -- instead of resolving it unconditionally at module
    # import time -- means --help, --reset-config, and first-run config
    # scaffolding (all handled above) work even when ComfyUI isn't
    # auto-detectable yet.
    from paths import get_comfy_dir, set_comfy_dir
    if config.paths.comfy_dir:
        # paths.comfy_dir in the TOML config is the highest-priority override
        # (higher than the COMFY_DIR env var) -- this previously did nothing
        # at all, since nothing ever read this field and passed it to
        # set_comfy_dir(). It was silently ignored no matter what you put here.
        set_comfy_dir(config.paths.comfy_dir)
    comfy_dir = get_comfy_dir()
    if str(comfy_dir) not in sys.path:
        sys.path.append(str(comfy_dir))

    from .trainer import Trainer
    trainer = Trainer(config)
    trainer.setup().load_models().build_optimizer().train()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

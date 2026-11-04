"""The experiment templates `python -m next.lab template KIND` prints: one per question asked so far, each with the arms that answer it.

The arguments of an older template are what was run, so they are not edited when the recipe moves on: ``LADDER_100`` is the
recipe of the region-count and span tests, ``RECIPE`` the current one."""

from __future__ import annotations

import re

LADDER_100 = ["--regions", "3", "--warmup", "50", "--regions-from", "9", "--regions-span", "100"]


def with_steps(template: dict, steps: int) -> str:
    """The template's training command with its step count and save interval (written once, at the end) replaced."""
    return re.sub(r"--save-every \d+", f"--save-every {steps}", re.sub(r"--steps \d+", f"--steps {steps}", template["train_command"]))


TEMPLATE = {
    "name": "range-ablation",
    "note": "Which part of the timestep range gives the painted-over look? Each arm trains on one range only. t is the timestep: t=0 is a clean image, t=999 is pure noise.",
    "train_command": "python -m next.run sdxl-lora --models /home/okolenmi/comfy/ComfyUI/models/ --datasets /home/okolenmi/Desktop/B580-diffusion-training/datasets/ --dataset my_set --file divA.safetensors --output x --batch 2 --resolution 512 --steps 600 --save-every 600 --set runtime.deterministic=true --set trainer.probe_every=100 --set trainer.probe_items=4",
    "output": "lab-out",
    "device": "xpu",
    "images": {"train": "/path/to/training/images", "holdout": "/path/to/images/not/in/training", "count": 6, "size": 768},
    "grid": {"prompts": ["", "a portrait of a woman, detailed face", "a red house on a green hill under a blue sky"], "seeds": [1, 2, 3], "strengths": [1.0],
             "steps": 20, "cfg": 6.0, "sampler": "euler", "scheduler": "normal", "width": 1024, "height": 1024},
    "arms": [{"name": "full"},
             {"name": "t0-250", "args": ["--set", "noiser.low=0", "--set", "noiser.high=250"], "note": "t 0-250 only (sigma 0.03-0.70, nearly clean images)"},
             {"name": "t250-600", "args": ["--set", "noiser.low=250", "--set", "noiser.high=600"], "note": "sigma 0.70-2.29"},
             {"name": "t600-999", "args": ["--set", "noiser.low=600", "--set", "noiser.high=999"], "note": "sigma 2.29-14.6 (mostly noise)"}],
}


STRATA = ["--set", "noiser.regions=3", "--set", "noiser.layout=strata"]
REGIONS_TEMPLATE = {
    **TEMPLATE,
    "name": "regions",
    "note": "Three timesteps per image and how their gradients are combined. `batch3` is the fair control: the same cost with three different images and one timestep each.",
    "train_command": TEMPLATE["train_command"].replace("--batch 2 --resolution 512", "--batch 1 --resolution 1024").replace("--steps 600", "--steps 300").replace("--save-every 600", "--save-every 300").replace(" --set runtime.deterministic=true", ""),
    "arms": [{"name": "single"},
             {"name": "batch3", "args": ["--batch", "3"]},
             {"name": "regions3-mean", "args": ["--set", "noiser.regions=3"]},
             {"name": "strata-equal", "args": [*STRATA, "--set", "trainer.gradient_policy=equal"], "note": "each region's gradient scaled to unit length, then averaged"},
             {"name": "strata-project", "args": [*STRATA, "--set", "trainer.gradient_policy=project"]},
             {"name": "strata-common", "args": [*STRATA, "--set", "trainer.gradient_policy=common"]}],
}
EQUAL = [*STRATA, "--set", "trainer.gradient_policy=equal"]
SIGNAL_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "regions-signal",
    "note": "Two questions. Does input perturbation (extra noise in the network input only, a different strength per region) help? "
            "And once each region's gradient is averaged over 4 batches (less noise), do the regions conflict, so that project or common differ from equal?",
    "arms": [{"name": "equal", "args": EQUAL},
             {"name": "equal-perturb", "args": [*EQUAL, "--set", "noiser.perturb=0,0.05,0.1"]},
             {"name": "equal-perturb-big", "args": [*EQUAL, "--set", "noiser.perturb=0,0.1,0.2"]},
             {"name": "equal-acc4", "args": [*EQUAL, "--accumulation", "4", "--steps", "100"], "note": "100 steps of 4 batches"},
             {"name": "project-acc4", "args": [*STRATA, "--set", "trainer.gradient_policy=project", "--accumulation", "4", "--steps", "100"]},
             {"name": "common-acc4", "args": [*STRATA, "--set", "trainer.gradient_policy=common", "--accumulation", "4", "--steps", "100"]}],
}
PERTURB_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "regions-perturb",
    "note": "Input perturbation by region. Region 0 is the nearly clean end of the range (t 0-333), region 2 the noisy end (t 666-999). "
            "Strength 0 at region 2 keeps the structure; the strength at region 0 is what is tried.",
    "arms": [{"name": "equal", "args": ["--regions", "3"]},
             {"name": "perturb-low", "args": ["--regions", "3", "--set", "noiser.perturb=0.2,0.1,0"]},
             {"name": "perturb-low-big", "args": ["--regions", "3", "--set", "noiser.perturb=0.4,0.2,0"]},
             {"name": "perturb-low-only", "args": ["--regions", "3", "--set", "noiser.perturb=0.3,0,0"]}],
}
CHEAP_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "cheap-equal",
    "note": "Can per-timestep weights (1 / the typical gradient norm of each third) give what three timesteps per image give, at a third of the cost? "
            "`equal` costs three forwards per step; `gradnorm-900` has the same compute with single-timestep steps.",
    "arms": [{"name": "equal", "args": ["--regions", "3"]},
             {"name": "gradnorm-300", "args": ["--time-balance", "gradnorm"], "note": "a third of the compute"},
             {"name": "gradnorm-900", "args": ["--time-balance", "gradnorm", "--steps", "900", "--save-every", "900"]},
             {"name": "single-900", "args": ["--steps", "900", "--save-every", "900"], "note": "same compute as equal, no weights"}],
}
NOISE_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "noise-floor",
    "note": "Same arguments, same seed, run twice (and twice more with deterministic kernels): how different are two runs that should be the same? "
            "Differences between arms in other experiments smaller than this are not results.",
    "train_command": with_steps(REGIONS_TEMPLATE, 150),
    "arms": [{"name": "equal-a", "args": ["--regions", "3"]},
             {"name": "equal-b", "args": ["--regions", "3"]},
             {"name": "det-a", "args": ["--regions", "3", "--set", "runtime.deterministic=true"]},
             {"name": "det-b", "args": ["--regions", "3", "--set", "runtime.deterministic=true"]}],
}
EPS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "eps-scale",
    "note": "Epsilon scaling at sampling time (the predicted noise is divided by the factor): does a small correction remove the painted look or the composition errors "
            "of an adapter that is already trained? Put the path of a trained adapter in the first arm; nothing is trained.",
    "arms": [{"name": "as-trained", "lora": "/path/to/lora_step000300.safetensors"},
             {"name": "eps1.005", "reuse": "as-trained", "eps_scale": 1.005},
             {"name": "eps1.01", "reuse": "as-trained", "eps_scale": 1.01},
             {"name": "eps1.02", "reuse": "as-trained", "eps_scale": 1.02},
             {"name": "eps1.04", "reuse": "as-trained", "eps_scale": 1.04}],
}
LR_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "lr",
    "note": "Learning rate, with the control run twice: the two controls show how far apart two identical runs end up, "
            "so a learning-rate effect smaller than that gap is not a result.",
    "arms": [{"name": "lr-1e-4", "args": ["--regions", "3"]},
             {"name": "lr-1e-4-again", "args": ["--regions", "3"]},
             {"name": "lr-3e-5", "args": ["--regions", "3", "--lr", "3e-5"]},
             {"name": "lr-3e-4", "args": ["--regions", "3", "--lr", "3e-4"]}],
}
ONE_PASS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "one-pass",
    "note": "Three timesteps per image with one forward and backward per step instead of three. `equal` runs each timestep alone; `mean` puts them in one batch "
            "without scaling; the calibrated arms run the regions alone only every Nth step (to measure each one's gradient norm) and as one weighted batch in between.",
    "arms": [{"name": "equal", "args": ["--regions", "3"]},
             {"name": "mean", "args": ["--regions", "3", "--set", "trainer.gradient_policy=mean"]},
             {"name": "calibrated-10", "args": ["--regions", "3", "--time-balance", "gradnorm", "--set", "trainer.policy_calibrate_every=10"]},
             {"name": "calibrated-50", "args": ["--regions", "3", "--time-balance", "gradnorm", "--set", "trainer.policy_calibrate_every=50"]}],
}
FIRST_STEPS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "first-steps",
    "note": "The first updates set the direction the adapter takes. Three ways to make them less random: a learning-rate warmup, and more timesteps per image at the "
            "start that fall to 3 over the first 100 steps (every region is a full pass, so the start is slow).",
    "arms": [{"name": "equal", "args": ["--regions", "3"]},
             {"name": "warmup-50", "args": ["--regions", "3", "--warmup", "50"]},
             {"name": "ladder-9", "args": ["--regions", "3", "--regions-from", "9", "--regions-span", "100"]},
             {"name": "ladder-6-warmup", "args": ["--regions", "3", "--warmup", "50", "--regions-from", "6", "--regions-span", "100"]}],
}
REGION_COUNT_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "region-count",
    "note": "How many timesteps per image: 2 and 4 against 3, and the ladder from 9. Every arm has the 50-step warmup; `ladder-6` is the best arm of the first-steps test, run again as the control.",
    "arms": [{"name": "ladder-6", "args": ["--regions", "3", "--warmup", "50", "--regions-from", "6", "--regions-span", "100"]},
             {"name": "ladder-9", "args": LADDER_100},
             {"name": "regions-2", "args": ["--regions", "2", "--warmup", "50"]},
             {"name": "regions-4", "args": ["--regions", "4", "--warmup", "50"]}],
}
LADDER_SPAN_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "ladder-span",
    "note": "How long the extra timesteps per image should last, and whether ladder 9 beating ladder 6 survives another data order. "
            "The control is `ladder-9` of the region-count test (600 steps, warmup 50, span 100, 3 regions): same settings, so its pictures are the reference.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "ladder-9-span-50", "args": ["--regions", "3", "--warmup", "50", "--regions-from", "9", "--regions-span", "50"]},
             {"name": "ladder-9-span-200", "args": ["--regions", "3", "--warmup", "50", "--regions-from", "9", "--regions-span", "200"]},
             {"name": "ladder-9-seed-2", "args": ["--regions", "3", "--warmup", "50", "--regions-from", "9", "--regions-span", "100", "--set", "batcher.seed=7"]}],
}
BATCH_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "batch",
    "note": "Different images in one pass. Every pass so far had one image; with 3 timesteps per image each region is a batch-1 pass, bound by kernel launches, so 2 or 4 images may cost much less than 2 or 4 times as long. "
            "The control is `ladder-9` of the region-count test (one image, same settings, 600 steps).",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "batch-2", "args": LADDER_100 + ["--batch", "2"]},
             {"name": "batch-4", "args": LADDER_100 + ["--batch", "4"]},
             {"name": "batch-4-lr-2e-4", "args": LADDER_100 + ["--batch", "4", "--lr", "2e-4"]}],
}
LONG_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "long-run",
    "note": "A long run with the best recipe so far, with a constant learning rate and with a cosine decay to 5% of it. Schedule-free optimizers match the loss but gave worse samples "
            "in a diffusion benchmark, which the authors put down to the missing learning-rate decay, so the decay is what is tried here.",
    "train_command": with_steps(REGIONS_TEMPLATE, 1200),
    "arms": [{"name": "constant-1200", "args": LADDER_100},
             {"name": "cosine-1200", "args": LADDER_100 + ["--schedule", "cosine"]}],
}
RECIPE = ["--regions", "3", "--warmup", "50", "--regions-from", "9", "--regions-span", "120"]
LORA_PLUS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "lora-plus",
    "note": "LoRA+: the up-projection matrix (B) learns faster than the down-projection (A). B starts at zero, so with one rate it lags behind; a ratio of 4 to 16 is the range "
            "the method's authors found to speed learning up. The baseline is the recipe of the long-run test (600 steps), the arms change only the ratio.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "baseline", "args": RECIPE},
             {"name": "lora-plus-4", "args": RECIPE + ["--lora-plus", "4"]},
             {"name": "lora-plus-16", "args": RECIPE + ["--lora-plus", "16"]}],
}
EMA_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "ema",
    "note": "Averaged adapter weights against the trained ones, from one run: train with `--ema 0.05,0.1` added to the recipe (it saves `lora_stepN_ema0.05.safetensors` and `_ema0.1` beside each file) "
            "and put the three paths below. Nothing is trained here.",
    "arms": [{"name": "as-trained", "lora": "/path/to/lora_step000600.safetensors"},
             {"name": "ema-0.05", "lora": "/path/to/lora_step000600_ema0.05.safetensors"},
             {"name": "ema-0.1", "lora": "/path/to/lora_step000600_ema0.1.safetensors"}],
}
ALIGNED_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "aligned-init",
    "note": "Gradient-aligned start of the adapter: before the first step the A factors are set to the main directions of the loss gradient measured on the first N batches (B stays zero, "
            "the model is unchanged). The baseline is the recipe of the lora-plus test; the arms differ in how many batches are measured (they are consumed, so training starts that far into the data and the region ladder).",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "baseline", "args": RECIPE},
             {"name": "aligned-4", "args": RECIPE + ["--aligned-init", "4"]},
             {"name": "aligned-16", "args": RECIPE + ["--aligned-init", "16"]}],
}
BATCH_RAMP_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "batch-ramp",
    "note": "Batch 1 at the start and a bigger batch at the end. Batch 2 costs the same time as batch 1, batch 4 about 1.7 times (batch test). Every arm has batch size 4 at the end and the learning rate 2e-4 "
            "that suited it; the controls are `batch-4-lr-2e-4` of the batch test (4 from the start) and the same recipe with batch 1 all the way. 600 steps.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "batch-1-lr-2e-4", "args": LADDER_100 + ["--batch", "1", "--lr", "2e-4"]},
             {"name": "ramp-1-to-4-at-400", "args": LADDER_100 + ["--batch", "4", "--lr", "2e-4", "--batch-first", "1", "--batch-first-steps", "400"]},
             {"name": "ramp-1-to-4-at-250", "args": LADDER_100 + ["--batch", "4", "--lr", "2e-4", "--batch-first", "1", "--batch-first-steps", "250"]},
             {"name": "ramp-2-to-4-at-400", "args": LADDER_100 + ["--batch", "4", "--lr", "2e-4", "--batch-first", "2", "--batch-first-steps", "400"]}],
}
TREND_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "trend",
    "note": "Trend weighting: each region of a step gets a weight from how its part of the timestep range is doing (12 smooth bins, a fast average of the loss against a slow one). "
            "A region doing better than usual counts more; nothing is taken away unless a penalty is set. `trend` reads the loss of the bins over the recent steps, `current` the region's own loss "
            "against its bins' average (noisier). The weights are applied after each region's gradient is scaled to unit length, so they change the direction and the length of the update. "
            "Step lines show `trend w` per region. 600 steps; `baseline` is the recipe of the long-run test.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "baseline", "args": RECIPE},
             {"name": "trend-1", "args": RECIPE + ["--trend-boost", "1"]},
             {"name": "trend-3", "args": RECIPE + ["--trend-boost", "3"]},
             {"name": "current-1", "args": RECIPE + ["--trend-boost", "1", "--trend-mode", "current"]},
             {"name": "trend-1-penalty", "args": RECIPE + ["--trend-boost", "1", "--trend-penalty", "0.5"]}],
}
TEMPLATES = {"ranges": TEMPLATE, "regions": REGIONS_TEMPLATE, "regions-signal": SIGNAL_TEMPLATE, "regions-perturb": PERTURB_TEMPLATE,
             "cheap-equal": CHEAP_TEMPLATE, "noise-floor": NOISE_TEMPLATE, "eps-scale": EPS_TEMPLATE, "lr": LR_TEMPLATE, "one-pass": ONE_PASS_TEMPLATE, "first-steps": FIRST_STEPS_TEMPLATE, "region-count": REGION_COUNT_TEMPLATE, "ladder-span": LADDER_SPAN_TEMPLATE, "batch": BATCH_TEMPLATE, "long-run": LONG_TEMPLATE, "lora-plus": LORA_PLUS_TEMPLATE, "ema": EMA_TEMPLATE, "aligned-init": ALIGNED_TEMPLATE, "batch-ramp": BATCH_RAMP_TEMPLATE, "trend": TREND_TEMPLATE}


TAKEN_FLAGS = ("--models", "--datasets", "--dataset", "--file", "--resolution")


def like(template: dict, earlier: dict) -> dict:
    """``template`` with the setup of an experiment file already filled in: images, grid, output folder, device, and the model,
    dataset and resolution arguments of its training command. Everything that defines the experiment (arms, steps, note) stays."""
    out = dict(template)
    for key in ("images", "grid", "output", "device"):
        if key in earlier:
            out[key] = earlier[key]
    command = out.get("train_command", "")
    for flag in TAKEN_FLAGS:
        match = re.search(rf"(?<=\s){re.escape(flag)}\s+(\S+)", earlier.get("train_command", ""))
        if match:
            command = re.sub(rf"(?<=\s){re.escape(flag)}\s+\S+", lambda _: f"{flag} {match.group(1)}", command, count=1)
    out["train_command"] = command
    return out

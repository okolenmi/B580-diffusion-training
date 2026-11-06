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
             {"name": "strata-equal", "args": [*STRATA, "--gradient-policy", "equal"], "note": "each region's gradient scaled to unit length, then averaged"},
             {"name": "strata-project", "args": [*STRATA, "--gradient-policy", "project"]},
             {"name": "strata-common", "args": [*STRATA, "--gradient-policy", "common"]}],
}
EQUAL = [*STRATA, "--gradient-policy", "equal"]
SIGNAL_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "regions-signal",
    "note": "Two questions. Does input perturbation (extra noise in the network input only, a different strength per region) help? "
            "And once each region's gradient is averaged over 4 batches (less noise), do the regions conflict, so that project or common differ from equal?",
    "arms": [{"name": "equal", "args": EQUAL},
             {"name": "equal-perturb", "args": [*EQUAL, "--set", "noiser.perturb=0,0.05,0.1"]},
             {"name": "equal-perturb-big", "args": [*EQUAL, "--set", "noiser.perturb=0,0.1,0.2"]},
             {"name": "equal-acc4", "args": [*EQUAL, "--accumulation", "4", "--steps", "100"], "note": "100 steps of 4 batches"},
             {"name": "project-acc4", "args": [*STRATA, "--gradient-policy", "project", "--accumulation", "4", "--steps", "100"]},
             {"name": "common-acc4", "args": [*STRATA, "--gradient-policy", "common", "--accumulation", "4", "--steps", "100"]}],
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
             {"name": "mean", "args": ["--regions", "3", "--gradient-policy", "mean"]},
             {"name": "calibrated-10", "args": ["--regions", "3", "--time-balance", "gradnorm", "--policy-calibrate-every", "10"]},
             {"name": "calibrated-50", "args": ["--regions", "3", "--time-balance", "gradnorm", "--policy-calibrate-every", "50"]}],
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
BEST = LADDER_100 + ["--batch", "1", "--lr", "2e-4"]
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
    "note": "Averaged adapter weights against the trained ones, from one run: `trained` trains the recipe with `--ema 0.05,0.1` (the trainer writes `lora_stepN_ema0.05.safetensors` and "
            "`_ema0.1` beside each file) and the other arms render those files. The averages are rendered from the same run, so the only difference is the averaging.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "trained", "args": RECIPE + ["--ema", "0.05,0.1"]},
             {"name": "ema-0.05", "reuse": "trained", "ema": 0.05},
             {"name": "ema-0.1", "reuse": "trained", "ema": 0.1}],
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
ALIGNED_VALUES_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "aligned-values",
    "note": "Gradient-aligned start, other values. In the first test 4 batches gave the best textures and 16 was a clear fail, so the arms bracket 4 (2 and 6), and 4 is combined with LoRA+ "
            "(a faster B suited the aligned A in theory: A is already a good direction, B has to grow from zero). `aligned-4` is the control from the first test, run again so every picture "
            "is on one sheet. Recipe and 600 steps as before.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "aligned-4", "args": RECIPE + ["--aligned-init", "4"]},
             {"name": "aligned-2", "args": RECIPE + ["--aligned-init", "2"]},
             {"name": "aligned-6", "args": RECIPE + ["--aligned-init", "6"]},
             {"name": "aligned-4-lora-plus-4", "args": RECIPE + ["--aligned-init", "4", "--lora-plus", "4"]},
             {"name": "aligned-4-lora-plus-8", "args": RECIPE + ["--aligned-init", "4", "--lora-plus", "8"]}],
}
CONTINUE_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "continue-best",
    "note": "Does more training improve a good adapter or damage it? `best` is your saved adapter (batch 1, lr 2e-4, 600 steps) rendered as it is; the other arms continue it (`--init-adapter`) "
            "with a fresh optimizer and a 50 step warmup, no region ladder (3 regions from the start). The same data order as the first run starts again, so the first images seen are the "
            "same ones. Compare each arm with `best` on the same sheet, and the per-timestep table (holdout above 1 means it got worse than the base model there).",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "more-300-lr-2e-4", "args": ["--regions", "3", "--warmup", "50", "--batch", "1", "--lr", "2e-4", "--init-adapter", "{lora:best}"]},
             {"name": "more-300-lr-5e-5", "args": ["--regions", "3", "--warmup", "50", "--batch", "1", "--lr", "5e-5", "--init-adapter", "{lora:best}"]},
             {"name": "more-600-lr-2e-4", "args": ["--regions", "3", "--warmup", "50", "--batch", "1", "--lr", "2e-4", "--steps", "600", "--save-every", "600", "--init-adapter", "{lora:best}"]}],
}
ANCHOR_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "anchor",
    "note": "Anchor loss: every second step the network is also asked about the batch's noisy images under a random prompt unrelated to the dataset (a built-in list of eight scenes and portraits), "
            "with and without the adapter, and the difference between the two predictions is penalised. The adapter keeps what it learns from the dataset but is held back from changing how "
            "the model answers other prompts. Look at the `portrait` and `red house` pictures (the dataset is not in them): do they keep the base model's look better with a stronger "
            "weight, and what is lost on the empty prompt? The baseline is your best recipe (batch 1, lr 2e-4, 600 steps); an anchor step costs about one more pass, so the arms take longer.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "baseline", "args": BEST},
             {"name": "anchor-1", "args": BEST + ["--anchor", "1"]},
             {"name": "anchor-5", "args": BEST + ["--anchor", "5"]},
             {"name": "anchor-25", "args": BEST + ["--anchor", "25"]}],
}
CLEAN_START = ["--regions", "3", "--batch", "1", "--init-adapter", "{lora:best}", "--clean-steps", "200"]
CLEAN_ONLY = ["--steps", "1", "--save-every", "1", "--lr", "1e-8"]
CLEAN_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "clean",
    "note": "LoRA cleaner on your saved best adapter. Every weight (`element`) or every singular direction of each layer (`component`) gets a gate; the adapter is frozen and only the gates train for 200 "
            "steps with the training loss, and with `--anchor` also the damage to other prompts; then the highest 15% stay. The `-only` arms stop right after the cleaning (one step at a tiny rate), "
            "so their pictures are the cleaned adapter itself: does the dataset's look survive in 15% of the weights, and do the unrelated-prompt pictures (portrait, red house) come closer to the base "
            "model than in `best`? `no-anchor` shows what the second signal adds, and `magnitude` is the control without any learning (the largest 15% of the weights stay): the gates are worth it only if they beat it. `more` continues for 300 steps with the kept part frozen, the cleared room training from zero. "
            "The cleaned adapters, the kept masks and a summary of what was kept per part of the network are in each arm's `train/clean/` folder.",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "element-15-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--anchor", "5"]},
             {"name": "element-15-no-anchor-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15"]},
             {"name": "magnitude-15-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "magnitude"], "note": "control: no gates, the largest 15% of the weights stay"},
             {"name": "component-15-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "component", "--anchor", "5"]},
             {"name": "element-15-more", "args": CLEAN_START + ["--lr", "2e-4", "--warmup", "20", "--clean", "0.15", "--anchor", "5"]}],
}
LORA_PLUS_VALUES_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "lora-plus-values",
    "note": "LoRA+ ratios closer together. Both 4 and 16 looked good in the first test, so the arms are 2, 4, 8 and 12 on the same recipe and 600 steps; `baseline` (no LoRA+) is run again so every "
            "picture is on one sheet.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "baseline", "args": RECIPE},
             {"name": "lora-plus-2", "args": RECIPE + ["--lora-plus", "2"]},
             {"name": "lora-plus-4", "args": RECIPE + ["--lora-plus", "4"]},
             {"name": "lora-plus-8", "args": RECIPE + ["--lora-plus", "8"]},
             {"name": "lora-plus-12", "args": RECIPE + ["--lora-plus", "12"]}],
}
CLEAN_LR_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "clean-lr",
    "note": "After the cleaning, how fast should the cleared room train? The kept 15% is frozen; the rest trains at a much lower rate than 2e-4 (which is the rate that produced the adapter and may overshoot in "
            "sensitive places). `best` and `element-15-only` are for the same sheet; the other arms clean (200 gate steps, the same as before) and then train 300 steps at the rate in the name.",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "element-15-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--anchor", "5"]},
             {"name": "more-lr-2e-5", "args": CLEAN_START + ["--lr", "2e-5", "--warmup", "20", "--clean", "0.15", "--anchor", "5"]},
             {"name": "more-lr-4e-6", "args": CLEAN_START + ["--lr", "4e-6", "--warmup", "20", "--clean", "0.15", "--anchor", "5"]}],
}
CLEAN_ANCHOR_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "clean-anchor",
    "note": "The cleaner with other anchor weights. In the first test anchor 5 removed nearly all of the dataset's effect with 15% of the weights, and no anchor kept most of it (about as much as the magnitude "
            "control). These arms are in between, all stopping right after the cleaning: element with anchor 0.5, 1 and 2, and component with 1. `best` is for the same sheet.",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "element-anchor-0.5-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--anchor", "0.5"]},
             {"name": "element-anchor-1-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--anchor", "1"]},
             {"name": "element-anchor-2-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--anchor", "2"]},
             {"name": "component-anchor-1-only", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "component", "--anchor", "1"]}],
}
CLEAN_MAGNITUDE_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "clean-magnitude",
    "note": "Variants of the cleaning that won the first test: keep the largest weights, no training (each arm is quick; the rendering is the cost). `magnitude-15` is the control of the first test for "
            "one sheet; 8 and 30 show how much stays at other shares; `layer` counts the 15% inside every layer instead of over the whole adapter (large layers no longer keep more); `product` ranks a "
            "weight by what it contributes to the layer's change (|weight| times the size of the other factor's row or column) instead of by itself; `rescale` scales every layer's change back to its "
            "old size after the cut (the lost power made up); `singular` keeps the largest singular directions of each layer's change (about 2 of 16) instead of single weights.",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "magnitude-15", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "magnitude"]},
             {"name": "magnitude-8", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.08", "--clean-mode", "magnitude"]},
             {"name": "magnitude-30", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.30", "--clean-mode", "magnitude"]},
             {"name": "magnitude-15-layer", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "magnitude", "--clean-scope", "layer"]},
             {"name": "product-15", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "product"]},
             {"name": "magnitude-15-rescale", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "magnitude", "--clean-rescale"]},
             {"name": "singular-15", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.15", "--clean-mode", "singular"]}],
}
CLEAN_PARTS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "clean-parts",
    "note": "Which parts of the adapter matter for what you see? Each arm is `best` with one part or region removed completely (99% of the rest stays): feed-forward, self-attention (attn1), cross-attention "
            "(attn2), the projections in and out of the transformer blocks, and the input, middle and output blocks. Nothing is trained; the pictures show what each part contributes and what the "
            "adapter does without it.",
    "train_command": with_steps(REGIONS_TEMPLATE, 300),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             *({"name": f"without-{label}", "args": CLEAN_START + CLEAN_ONLY + ["--clean", "0.99", "--clean-mode", "magnitude", "--clean-drop", drop]}
               for label, drop in (("ff", "ff"), ("attn1", "attn1"), ("attn2", "attn2"), ("proj", "proj_in,proj_out"),
                                   ("input-blocks", "input_blocks"), ("middle-block", "middle_block"), ("output-blocks", "output_blocks"))
               )],
}
TARGETS_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "targets",
    "note": "Train without the parts that did nothing for the loss and made the pictures worse when the saved best adapter was cut (item 69). The recipe is your best one (batch 1, lr 2e-4, 600 steps); `best` is "
            "your saved adapter for the same sheet. `no-attn2`: no cross-attention (the captions are empty, so it only learns to misread one constant vector). `no-middle`: no adapter in the middle block. "
            "`no-attn2-no-middle`: both. `no-tier-2`: both plus input blocks 6-8 and output blocks 0-2 (the second most sensitive tier by your estimate) left out.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "no-attn2", "args": BEST + ["--param", "adapter.groups=attn_self,ff,transformer_proj"]},
             {"name": "no-middle", "args": BEST + ["--param", "adapter.exclude=middle_block\\..*"]},
             {"name": "no-attn2-no-middle", "args": BEST + ["--param", "adapter.groups=attn_self,ff,transformer_proj", "--param", "adapter.exclude=middle_block\\..*"]},
             {"name": "no-tier-2", "args": BEST + ["--param", "adapter.groups=attn_self,ff,transformer_proj", "--param", "adapter.exclude=middle_block\\..*|input_blocks\\.[678]\\..*|output_blocks\\.[012]\\..*"]}],
}
def stages(*specs: str) -> list[str]:
    """The arguments of a staged run; the total steps are given too so that the report counts them."""
    total = sum(int(spec.partition(":")[0].partition("@")[0]) for spec in specs)
    return [*BEST, "--steps", str(total), *(arg for spec in specs for arg in ("--stage", spec))]


TIER_2 = ("input_blocks.6", "input_blocks.7", "input_blocks.8", "output_blocks.0", "output_blocks.1", "output_blocks.2")


def block_rates(middle: float, second: float) -> list[str]:
    """Your sensitivity tiers as rate factors: the middle block, then input blocks 6-8 and output blocks 0-2."""
    return [*BEST, "--lr-region", f"middle_block={middle:g}", *(arg for prefix in TIER_2 for arg in ("--lr-region", f"{prefix}={second:g}"))]


BLOCK_RATES_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "block-rates",
    "note": "A learning rate per part of the network, from your sensitivity estimate: the middle block is the most sensitive, input blocks 6-8 and output blocks 0-2 the second tier, the rest not much. "
            "The recipe is your best one (batch 1, lr 2e-4, 600 steps); `best` is your saved adapter for the same sheet. `mild`: middle x0.5, second tier x0.75. `tiers`: middle x0.25, second tier x0.5. "
            "`strong`: middle x0.1, second tier x0.3. `middle-only`: only the middle block at x0.25.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "mild", "args": block_rates(0.5, 0.75)},
             {"name": "tiers", "args": block_rates(0.25, 0.5)},
             {"name": "strong", "args": block_rates(0.1, 0.3)},
             {"name": "middle-only", "args": [*BEST, "--lr-region", "middle_block=0.25"]}],
}
STAGES_TEMPLATE = {
    **REGIONS_TEMPLATE,
    "name": "stages",
    "note": "Staged training in one run: train, clean to a small share of the largest weights, freeze them, train the rest at a lower rate, clean to a larger share, and so on (`--stage STEPS[@LR][:KEEP]`; "
            "KEEP is the frozen share of the whole adapter in total, the rate of a stage after the first ramps up over 20 steps). The first stage is your best recipe (lr 2e-4). `best` is your saved 600-step "
            "adapter for the same sheet. `8-15-30`: 600 steps, cleaning to 8%; 300 steps at 5e-5, to 15%; 300 steps at 2e-5, to 30%. `8-15`: the first two stages only. `8-slow`: cleaning to 8% then 600 steps at 2e-5. "
            "`15-30`: the same idea starting from 15%. Every arm trains, so each takes 1-1.5 hours; pick with `--arms`. The cleaned adapter of every stage is in the arm's `clean` folder.",
    "train_command": with_steps(REGIONS_TEMPLATE, 600),
    "arms": [{"name": "best", "lora": "{lora:best}"},
             {"name": "8-15-30", "args": stages("600:0.08", "300@5e-5:0.15", "300@2e-5:0.30")},
             {"name": "8-15", "args": stages("600:0.08", "400@5e-5:0.15")},
             {"name": "8-slow", "args": stages("600:0.08", "600@2e-5")},
             {"name": "15-30", "args": stages("600:0.15", "400@5e-5:0.30")}],
}
TEMPLATES = {"ranges": TEMPLATE, "stages": STAGES_TEMPLATE, "block-rates": BLOCK_RATES_TEMPLATE, "targets": TARGETS_TEMPLATE, "clean-magnitude": CLEAN_MAGNITUDE_TEMPLATE, "clean-parts": CLEAN_PARTS_TEMPLATE, "clean-anchor": CLEAN_ANCHOR_TEMPLATE, "clean-lr": CLEAN_LR_TEMPLATE, "lora-plus-values": LORA_PLUS_VALUES_TEMPLATE, "clean": CLEAN_TEMPLATE, "anchor": ANCHOR_TEMPLATE, "aligned-values": ALIGNED_VALUES_TEMPLATE, "continue-best": CONTINUE_TEMPLATE, "regions": REGIONS_TEMPLATE, "regions-signal": SIGNAL_TEMPLATE, "regions-perturb": PERTURB_TEMPLATE,
             "cheap-equal": CHEAP_TEMPLATE, "noise-floor": NOISE_TEMPLATE, "eps-scale": EPS_TEMPLATE, "lr": LR_TEMPLATE, "one-pass": ONE_PASS_TEMPLATE, "first-steps": FIRST_STEPS_TEMPLATE, "region-count": REGION_COUNT_TEMPLATE, "ladder-span": LADDER_SPAN_TEMPLATE, "batch": BATCH_TEMPLATE, "long-run": LONG_TEMPLATE, "lora-plus": LORA_PLUS_TEMPLATE, "ema": EMA_TEMPLATE, "aligned-init": ALIGNED_TEMPLATE, "batch-ramp": BATCH_RAMP_TEMPLATE, "trend": TREND_TEMPLATE}

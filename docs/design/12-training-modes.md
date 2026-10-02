# Training modes: what exists, what does not, and what each would take

*[← design index](README.md)* · see also
[`11-core-removal.md`](11-core-removal.md)

`nodes/config_model.py` declares four tuning methods. **One is
implemented.** The other three are configuration the API accepts and the
UI offers, with no trainer behind them.

| `tuning.method` | Status | Where the maths lives |
|---|---|---|
| `lora` | **implemented** | `nodes/train/supervised.py`, `nodes/train/managed.py` |
| `cyclic` | not implemented | `archive/core/trainer.py` |
| `distillation` | not implemented | `archive/core/trainer.py` |
| `full` | not implemented | `archive/core/trainer.py` |

Nothing silently falls back. `nodes/` is reachable only through a graph
execution, and a graph is built from the nodes that exist, so there is no
path that accepts these three and quietly trains something else. The
archived trainer can still be run by hand
(`python -m archive.core.cli`) while it remains runnable.

## Why this is the state rather than a bug

The `nodes/` pipeline was built to supersede `core/trainer.py`, and it
does -- for supervised LoRA. It was never extended to the other three.
`nodes/train/supervised.py` says so in its own first paragraph: "no
gradient accumulation, no cyclic/teacher-rollout caching".

Implementing the other three is a project, not a config change, and
`core/` was archived rather than deleted so the reference maths is still
readable while that is decided.

## What each one needs

**`full` -- full fine-tuning rather than a LoRA adapter.** The smallest
of the three. `nodes/` has every piece except the actual thing: the
optimizer handles, the step pipeline, the schedule, the checkpoint
writer all operate on a parameter set, and a LoRA run is just one that
restricts the trainable subset. The work is deciding what "full" means
here -- which parameters, at what precision, and with what VRAM budget,
since an SDXL UNet in fp32 plus AdamW second moments does not fit in
12 GB. `scripts/hw_validate.py` is the tool for answering that on
hardware rather than by arithmetic. It is also the mode most likely to be
reachable, because it needs no new *maths*.

**`cyclic` -- alternate between two models within a run.** Needs a
two-model step loop and a cycle schedule, neither of which exists in
`nodes/train/`. `archive/core/trainer.py` has both, and
`manager/builder.py::DataTaskRunner.run_teacher_task` still contains the
teacher-rollout trajectory machinery it feeds on (still imported from
`nodes/components/noise_schedule.py` and `model_io.py`, which were kept
live precisely because that path exists). This is the cheapest of the
three in the sense that most of its parts are already written -- and the
most expensive in the sense that it needs a supervisor that owns two
lives at once, and a stop that means something different depending on
which one you interrupt.

**`distillation` -- train a student to match a teacher's prediction.**
The largest. It needs a teacher forward pass per student step, a
teacher/student type pairing (`vpred`/`eps`, and the conversions between
them), and a cache of teacher trajectories so the teacher is not run
every step. `manager/builder.py::run_teacher_task` builds those
trajectories today and its output is a dataset; the missing piece is a
trainer that consumes one. `nodes/components/noise_schedule.py` and
`model_io.py` exist to make that reachable without `core/`.

## If one of these is wanted

The order that makes sense is `full`, then `distillation`, then `cyclic`
-- by increasing trainer complexity. Each is a new `TrainerNode` beside
`SupervisedLoRATrainerNode`, plus a graph node to expose it, and the
config vocabulary already carries the settings. None of it requires
touching `nodes/components/`, which is the point of having moved the
config model and the diffusion helpers out of `core/` first.

Before writing any of them, measure. The Adafactor question in
`docs/known-issues/open.md` is the standing example on this machine: the
answer decides whether a `TinyBatchedStrategy` is worth writing at all,
and nobody knows it without running the two runs.
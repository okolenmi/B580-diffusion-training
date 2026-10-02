# Core removal: what is actually left

*[← design index](README.md)* · see also [`core-inventory.md`](../core-inventory.md)

**Status: not started.** Recorded 2026-10-02 because the investigation
that cleared the biggest question is expensive to repeat and its result is
easy to get backwards.

## The question that was open

"Is `core/` essential, or is it a route to something that already exists?"

**Answered by measurement, not by reading:**

| probe | result |
|---|---|
| Import every `nodes/` module with `core/` made **unimportable** | **96 of 96 succeed, 0 failures** |
| `smoke_test_managed_trainer.py` with `core/` unimportable | **passes** — real forward, real backward, real `ComposedAdamWOptimizerHandle`, real `optimizer.step()`, LoRA written to `.safetensors` |
| lazy (function-level) `core` imports inside `nodes/` | none |

`nodes/train/node.py` is the step loop; `nodes/train/managed.py` is the
pipeline. **The rewrite trains without the old trainer.** `core/` is not
essential to training, and calling it "the trainer" — which this repository
did, including in `docs/core-inventory.md` — was an inference from "the
backend launches `python -m core.cli`", which is a fact about which binary
runs, not about who does the work.

## Why it cannot simply be deleted

Five dependencies remain, in descending order of how much work they are.
None of them is the training logic.

**1. `nodes/` has no entry point.** No `main`, no `if __name__` block, no
CLI. It is a library that can train and nothing that can be launched — and
the backend supervises a *subprocess*, so something has to be runnable.
Today that is `core/cli.py`.

This is the one piece of genuinely new code. The trainer, the optimizers,
the step pipeline and the saving all exist and are proven by the probe
above; what is missing is a driver that reads the config, builds the
graph, runs the loop, writes `log.progress.jsonl` in the shape
`JsonlProgressSource` reads, and honours stop. The output format is fixed
by an existing consumer, which is most of the specification.

**2. `TrainingConfig` lives in `core/config_model.py`** (321 lines) and is
imported by four `backend/` modules: `domain/value_objects.py`,
`infrastructure/core_config_inspector.py`, `infrastructure/core_config_files.py`
and `infrastructure/config_schema.py` — the last of which derives the whole
settings UI from it. Notably `nodes/` does **not** import it (the two
`nodes/train/` mentions are docstrings, not imports), so moving it is a
backend-side change and does not constrain the trainer. It belongs
somewhere the backend owns.

**3. `manager/builder.py` imports 8 `core/` modules** for dataset
ingestion — `lora`, `model_io`, `noise_schedule`, `seed`, `unet_wrapper`,
`vae_decode`, `comfy_setup`, `clip_encode`. `nodes/` already has
equivalents for all eight, so this is mostly repointing, but it is on the
path of every run that has data, so it wants real tests rather than a
find-and-replace.

**4. Four `backend/` bridges** — `config_io`, `config_model`,
`comfy_setup`, `xpu_env`. Three fall out of (2); `comfy_setup.xpu_empty_cache`
is a genuine hardware helper and needs a decision about where it lives.

**5. Nine equivalence smoke tests** import `core/` as the reference
implementation. These are the reason `core/` must outlive the rest: they
are the only thing that can *demonstrate* the rewrite computes the same
numbers. They should be retired deliberately, once someone decides the
rewrite no longer needs proving — not deleted as collateral.

## Suggested order

2 → 1 → 3 → 4, with 5 last and deliberate. Moving the config model first
removes the constraint that would otherwise shape the trainer's interface;
the entry point is then unblocked; `manager/` follows; the bridges resolve
with (2) and (3).

## The rule worth keeping

Delete `core/` when nothing needs it, not when it is inconvenient. The
failure mode to avoid is the one this repository already made once: reading
"the backend launches `core.cli`" as "`core/` is the trainer", and then
building on it. Each of the five dependencies above is a *specific,
nameable* edge. When the list is empty, the folder goes. Until then it is
not dead code, and calling it legacy would be the misleading label again.
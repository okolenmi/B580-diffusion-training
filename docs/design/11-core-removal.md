# Core removal: what is actually left

*[← design index](README.md)* · see also [`core-inventory.md`](../core-inventory.md)

**Status: done, 2026-10-02.** `core/` is at `archive/core/`. Recorded
beforehand (and kept verbatim below) because the investigation that
cleared the biggest question is expensive to repeat and its result is
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

## What each edge became

| # | Edge | Resolution |
|---|---|---|
| 2 | `TrainingConfig` in `core/config_model.py` | Moved to `nodes/config_model.py`. Not to `backend/`, which was the plan: `backend/` depends on `nodes/` and never the reverse, and the trainer entry point needs the config too -- so a backend-owned module would have been unreachable from `nodes/`. Self-contained (pydantic + stdlib only). |
| 4 | `core/config_io` | Moved beside it to `nodes/config_io.py`. |
| 4 | `core/comfy_setup` (`xpu_empty_cache` etc.) | Deleted. `nodes/components/device.py` already had `DeviceContext.for_device("xpu").empty_cache()` -- same `hasattr(torch, "xpu")` guard, and `smoke_test_device_context_equivalence.py` proves the equivalence. The four call sites now use it. |
| 4 | `core/xpu_env` | Moved to `nodes/xpu_env.py`; called from `backend/cli.py` as it was from `core/cli.py`. |
| 3 | `manager/`'s eight imports | Three were already shims onto `nodes/` (`clip_encode`, `unet_wrapper`, `seed`) and now point there directly. `core/noise_schedule.py` and `core/model_io.py` became `nodes/components/noise_schedule.py` and `model_io.py` -- thin adapters over `nodes/components/diffusion.py`, whose docstrings already claimed to match them, verified bit-identical here. `core/vae_decode.py` moved to `nodes/model/vae_decode.py`; `nodes/` had no VAE decoder and `manager/`'s LoRA ingestion needs one. `T_MODES` stopped being a documented duplicate of `nodes/dataset/timestep_modes.py` and is now imported. |
| 5 | nine equivalence smoke tests | Not retired. They now import `archive.core.*`, which is the better answer to "retire them deliberately": they are the only evidence the rewrite computes the same numbers, and `core/` being archived is what lets them keep running. |
| 1 | no `nodes/` entry point | **Not done, and not needed.** Removing the backend's support for `core/` meant removing the route that spawned `python -m core.cli` -- the supervised-subprocess route and its whole run/supervision/reconcile stack -- not writing a replacement trainer. Training is now a graph execution, which `nodes/` already implements. A `nodes/cli.py` would be a new driver for a route that no longer exists. |

### The route that went with it

`POST /runs`, `POST /runs/{id}/stop`, `GET /runs`, `GET /runs/active`,
`GET /runs/{id}`, `GET /runs/{id}/log`, `DELETE /runs` and
`GET /config/start-options` are all gone, and with them `Run`, its
repository, the supervisor, the monitor, `start/stop_training`,
`reconcile_runs`, `RunLifecycleWriter`, the JSONL progress reader, the
seven `run_*` domain events, the dashboard (`views/dashboard.js`), the
run detail page, and the run-shaped value objects (`RunStatus`,
`StartFrom`).

What that costs, stated plainly: the dashboard no longer starts or stops
a run. Training is started from the graph editor, and execution history
lives at `/executions`. Two consequences worth knowing:

* `run_progressed` was the only **state** event, so
  `application/event_delivery.py`'s state class is now empty and
  nothing coalesces in production. The class and the buffer's
  overflow-ordering stay, because a delta must still be distinguishable
  from a lifecycle event.
* The `RunRepository` contract test went with it. Its value was that two
  implementations of one port must agree -- which is how WP-13 found the
  in-memory fake's drift -- and no port has two implementations any
  more. The CAS rule it partly covered is still pinned by
  `test_graph_execution.py` (six `update_if_status` sites).

## Why it could not simply be deleted

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
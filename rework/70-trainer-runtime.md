# 70. One trainer, a runtime layer, and the Diffusion and Optimization parts

Status: **proposal.** Depends on 10 (events), 20 (settings), 40 (family
`objective` and `memory` kits), 42 (conditioner), 43 (adapted parts), 50
(batcher, noiser). This is the plan that retires the two-trainer problem from the
component map.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What exists (audit-level facts)

* **Two trainers with separate loops.** `SupervisedLoRATrainerNode` (361 lines)
  + `step_pipeline.py` (750 lines, 10 phase classes) and `ManagedLoRATrainerNode`
  (`train/managed.py`, 1,714 lines, 15 classes, 31 ports) which re-implements
  FetchBatch, PrepareDiffusionInputs, EncodeConditioning, Forward, Loss,
  Monitoring and adds ZeroGrad, BackwardAndOptimizerStep, GraphForwardLoss...,
  ProbePhase and `AdaptiveResidencyController`.
* **The managed trainer does five jobs besides training**: residency (offload
  text encoder and optimizer around steps, calibrate on real peaks), text-encoder
  prewarm, periodic saving, probes (`t_probe.py`), graph capture
  (`xpu_graph_step.py`).
* **Phases are shared in spirit but not in code**; fused vs non-fused optimizers
  explain two different phase orderings (`OptimizerBeginStep/Backward/OptimizerStep`
  vs `ZeroGrad/BackwardAndOptimizerStep`).
* **Handles with no producer** (component map): `DiffusionProcess`,
  `Parameterization`, `AdapterStrategy`, `LoRAScalingPolicy`,
  `frozen_weight_store`, `ParameterGroupPolicy`.
* The LoRA timestep gate lives in a **module-level global** (`_current_gate` in
  `lora.py`, set per step by `PrepareDiffusionInputsPhase`; the code itself
  flags the limitation).
* Memory: `nodes/memory/` (coordinator, manager, device_reservations,
  graph_memory, control_handle), `VRAMBudgetControllerNode`, backend admission
  and ledger (ADR 0005).

## 2. Target shape

```
 batches ─┐
 parts ───┤ (adapted denoiser [+ encoders], multi-input)
 conditioner ─┤
 optimizer ───┼──► Trainer (the step loop only) ──► adapted parts (trained)
 lr_schedule ─┤                │
 objective ───┤                ├── emits: StepFinished, RunFinishing  (events, plan 10)
 loss_weighting ┘              └── uses:  ctx.runtime  (residency, memory, capture, profiler)
```

* **One `DiffusionTrainerNode`.** About seven handle ports (`batches`, `parts` (multi),
  `conditioner`, `optimizer`, `lr_schedule`, `objective`, `loss_weighting`) and
  two fields (`steps`, `grad_accum`). The 31-port node disappears; the 21-port
  supervised node too. The old nodes are not ported (plan 90).
* **A step engine library** (`engine/step/`): `StepState`, `StepPhase`,
  `Pipeline`, `TimedPhase`, and **one** implementation of each phase.
  The managed/supervised differences become **strategies injected into the
  phases**, not copies of them.
* **A runtime layer** (`engine/runtime/`) owning everything about *where things
  live and what the machine can bear*.
* **Outputs are subscribers**, not trainer ports (section 5).

## 3. The step engine: one set of phases

Phases (fixed order), each family-agnostic:

1. `FetchBatch`: next batch from `batches` (epoch events come from the batcher,
   plan 50).
2. `PrepareInputs`: move `x_t`, `t`, `target` to the device. The *formula* is
   already applied by the `Noiser` in the batch worker (plan 50), so this phase no
   longer knows eps from v from velocity.
3. `EncodeConditioning`: `conditioner.encode(prompts, sizes)`; residency policy
   wraps it (load encoder, encode, unload) only when the prompts are not cached.
4. `ZeroGrad` / `OptimizerBegin` through the `OptimizerHandle` strategy
   (`begin_step`, `after_backward`, `step`), so fused and non-fused share one
   phase list.
5. `Forward`: `family.denoiser.forward(adapted, x_t, t, conditioning)`.
6. `Loss`: `objective.loss(pred, target, t)` times `loss_weighting`.
7. `Backward` (+ optimizer step for fused).
8. `Publish`: emit `StepFinished(step, total, metrics)` and feed the display
   series (the old `MonitoringPhase`).
9. `Capture` variant: `runtime.graph_capture` selects a phase that replays
   forward+loss+backward through the XPU graph (`xpu_graph_step.py` stays; its
   shape-stability requirements are a family/batcher concern, noted in risks).

`Completion` (Stop keeps what ran) and cooperative cancel are unchanged.

## 4. The runtime layer (`ctx.runtime`)

Configured by graph settings (plan 20), not by ports.

| Service | Job | Replaces |
|---|---|---|
| `ResidencyManager` | knows every `DeviceResident` (parts, encoders, optimizer state), their offloadable flag and footprint; decides what is on the device now; calibration on real peaks, safety margin, escalation | `AdaptiveResidencyController`, `ResourceControlHandle` wiring, `VRAMBudgetControllerNode`, the residency ports |
| `Memory` | the graph's budget (`GraphMemory`, exists) and the ledger bridge | `resource_control` port |
| `BlockSwap` | holds N denoiser blocks on the host, brought in per block, using `family.memory.blocks()` | new (needed for Anima; the sd-scripts guide's main lever) |
| `Capture` | graph capture on/off and its constraints | `use_xpu_graph` port |
| `Profiler` | phase timings, per-phase memory | `profile`, `profile_memory_per_phase` |

* Residents **register themselves** when created (a loaded part, an encoder, the
  optimizer), through the existing `DeviceResident` interface; the trainer does not
  list them.
* The step engine asks the runtime, it does not manage residency inline. This is
  what removes the `empty_cache_every_n_steps` / prewarm / "residency escalation"
  tangle from the trainer's ports and puts it where plan 42's incremental prewarm
  can coordinate with it.
* **Prewarm** becomes: on `RunStarted` and `PromptsChanged`, the conditioner
  warms what it needs, then the runtime releases the encoders. Because both are
  in one place, the OOM reported earlier (cache emptying disabled) has one owner
  and one test.

## 5. Outputs are event subscribers

| Today (trainer port) | After |
|---|---|
| `save_every_n_steps`, `save_prefix` | `Every N Steps` trigger -> `CheckpointDue` -> `SaveAdapter` (event-in `save`) |
| `probe_every_n_steps`, `probe_items`, `probe_points_per_bucket`, `probe_grad_alignment` | `Probe` node (the `t_probe.py` logic) with event-in `run`; its four settings become its own fields |
| `monitor`, `on_step` | removed; displays come from `MODULES`, events from `StepFinished` (plan 60) |
| `profile*` | graph settings |

`Probe` and `Preview` need read access to the live model and a few batches. The
`StepFinished` payload may carry a read-only `TrainView` (an in-process object,
the one exception to "payloads are small", documented in plan 10); the trigger
passes it along in `run`. The saver already handles a partial result through
`Completion`.

Six trainer ports and two probes leave the node. **Lose nothing**: each existing
behaviour is reproduced by wiring, and the starters (plan 60) ship the wiring.

## 6. The Diffusion part (and the loose handles)

`ObjectiveNode` (new), grouped fields:

| Group | Fields | Output |
|---|---|---|
| **Process** | `kind` (choices from `family.objective`: ddpm schedule or flow), schedule options (shift, zero-terminal-SNR for ddpm) | `DiffusionProcess` |
| **Prediction** | `parameterization` (choices from the family: eps, v for SDXL; velocity for Anima) | `Parameterization` |
| **Timesteps** | policy (`uniform`, `range` with `t_low`/`t_high`, `values`, `sigmoid`, `shift`), seed | `TimestepSampler` |

Outputs: `objective` (consumed by the trainer) and `noiser` (consumed by the
batcher, plan 50). So **two of the six loose handles** get a producer. The other
four need no new nodes:

* `AdapterStrategy`, `LoRAScalingPolicy`, `frozen_weight_store` become **fields**
  on `InjectAdapterNode` (kind, scaling, weight store; plan 43); the handle types
  remain internal.
* `ParameterGroupPolicy`: produced by the adapted part (`param_groups()`, plan 43).

`LossWeightingNode`s (Uniform, Min-SNR, P2 exist; Anima adds `sigma_sqrt`,
`cosmap`): the choices come from `family.objective.loss_weights()`.

The **LoRA timestep gate** moves into the adapted part: the adapter applies the
gate from the `t` it receives; the module-level global disappears (it is a
documented bug waiting for a second trainer in one process, and plan 10's events
make that more likely).

## 7. The Optimization part

* **One node per algorithm** (AdamW, Adafactor, CAME) with a `fused` field; the
  trainer accepts either handle (today the choice is two classes per algorithm,
  six nodes). Group **Update rule**: learning rate, betas/eps, weight decay,
  `grad_clip_max_norm`; group **Parameters** (advanced): per-group multipliers
  consumed from `param_groups`.
* LR schedule nodes stay (Constant, Warmup, Cosine).
* `grad_accum` stays a loop field on the trainer; `grad_clip_max_norm` moves to
  the optimizer (it is an update rule).
* Future (cheap to add later, mentioned so the structure allows it): EMA of
  adapter weights as a subscriber of `StepFinished`; LoRA+ ratio via
  `param_groups`.

## 8. Conditioner and blend

`ConditioningBlendNode` takes a `conditioner` and returns a **wrapped conditioner**
that blends after the cached encode (as the current patch does after the cache,
so a random blend is never cached). The trainer therefore has **no**
`conditioning_blend` port.

`BucketBalance` feeds back from losses: today the trainer hands per-bucket losses
to it. Under events it is an **event-in** on `StepFinished` and its handle is read
by the bucketer; the trainer no longer knows about it.

## 9. Verification (replaces the earlier parity gates)

Per plan 90 there is no old-versus-new parity chain. What remains:

1. **One acceptance run**: the SDXL starter graph trains 5 to 10 steps on the
   synthetic fixture through the headless runner and saves a loadable adapter;
   events fire in order.
2. **Optional math spot checks** while the old trainer still exists: per-step loss
   of one step for a fixed seed and tiny denoiser, old versus new.
3. **Residency scenario tests** with a fake device (OOM-on-prewarm repro, offload
   and reload, calibration) are worth keeping: that code is the most fragile and
   the least visible from a loss curve.
4. **The owner's B580 run**: step time, peak memory and a short loss curve on the
   starter. Nothing in the sandbox can do this; the plan is not done without it.

## 10. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| R1 | Trainer count | two; one with strategies | One; the old two are not ported |
| R2 | Residency | trainer-internal; runtime service from graph settings | Runtime service |
| R3 | Saving and probes | trainer ports; event subscribers | Event subscribers |
| R4 | Diffusion producers | six nodes; one `ObjectiveNode` plus fields | One node plus fields |
| R5 | LoRA gate | global; owned by the adapted part | Owned by the adapted part |
| R6 | Blend | trainer port; conditioner wrapper | Conditioner wrapper |
| R7 | `BucketBalance` | trainer port; event subscriber | Event subscriber |
| R8 | Where the engine lives | `nodes/train/`; new top-level `engine/` | `engine/` (peer of `families/`), nodes stay thin |
| R9 | Fused optimizer handling | two classes; one handle with `begin_step/after_backward/step` | One handle strategy |

## 11. Risks

* **Performance**: an extra indirection per phase and opaque `Conditioning`.
  Mitigation: a step benchmark on CPU and the owner's hardware run; skip event
  dispatch when a topic has no subscriber.
* **Graph capture** needs static shapes; flow-matching 5-D latents and bucketing
  change that picture. Mitigation: capture stays opt-in, and the family declares
  whether its step is capture-safe.
* **Block swap is new code** and Anima may not fit without it; it must be measured
  on the B580, not assumed.
* **Residency regressions** (the earlier OOM). Mitigation: scenario tests with a
  fake device, and a "residency trace" display.
* **Scope**: this is the largest refactor. Mitigation: it is last in the order
  (plan 90), after families, Load Model, events and settings are in.

## 12. Open questions for the owner

1. Is a single trainer acceptable as the only training node (the old two are not
   ported)?
2. `grad_clip` on the optimizer and `grad_accum` on the trainer: agree?
3. Optional: recent B580 numbers for the current managed route (step time, peak MB)
   give the final acceptance run something to compare to. No longer a gate.

*[← docs/design index](README.md)*

# 7. Deliberately deferred or rejected

Considered, left out on purpose -- listed with the actual reasoning, not
just "future work":

- **`AutoResourcePolicy` (inspect available VRAM, decide strategies
  automatically).** `ResourcePolicy` (2.2) is designed so this is
  *possible* to add later without touching anything else -- but designing
  it *now*, in detail, would mean guessing at a real, hard ML-systems
  heuristic problem (predicting a training step's peak VRAM from param
  counts + batch shape + checkpointing granularity, ahead of actually
  running it) with no real-hardware data to validate against. `nodes/`'s
  own existing rule -- equivalence-test before switching over -- can't be
  followed for a heuristic with nothing to compare it against yet. Left
  as an interface-shaped placeholder, not a designed algorithm.
- **Automatic eviction inside `MemoryManager`.** Already considered and
  rejected once, correctly, in the existing module docstring ("no
  automatic eviction under memory pressure... behavior stays predictable
  rather than depending on runtime memory conditions"). Nothing here
  changes that reasoning; an `OffloadOrchestrator`-driven, *event*-
  triggered offload (5.2) is a different thing from pressure-triggered
  eviction inside the allocator itself, and doesn't need the latter.
- **Layer-wise CPU offload of the frozen UNet base between steps**
  (ZeRO-Infinity-style). Real, large VRAM lever -- not designed here: no
  existing per-block streaming-offload primitive to build on, and the
  PCIe round-trip cost per step is a real, hardware-dependent question
  that needs actual measurement, not an interface guess. `DeviceResident`
  leaves room for a future `LayeredOffload` variant of `TrainableModel`
  without foreclosing it.
- **Continuous-time / flow matching** (Lipman et al., arXiv:2210.02747;
  Liu et al.'s rectified flow, arXiv:2209.03003 -- the formulation Stable
  Diffusion 3 and Flux actually train with). `NoiseSchedule`/
  `Parameterization` (1.4) already leave the seam open (a sibling
  `Interpolant` contract; `convert_to()` already generalizes to a third,
  velocity-target `Parameterization`). Not designed further because it's
  not a drop-in swap for this project's actual model: SDXL is a
  pretrained epsilon/v-prediction model, and converting an already-
  trained diffusion model's *sampling trajectory* into a flow-matching
  one is itself an active, nontrivial research question -- Schusterbauer
  et al.'s "Diff2Flow" (CVPR 2025) exists specifically to do this
  alignment, which wouldn't be a real research topic if it were simple.
- **GaLore** (Zhao et al., arXiv:2403.03507, ICML 2024) -- projects
  gradients into a low-rank subspace so a full-parameter optimizer's
  state costs close to what LoRA's optimizer state already costs, without
  restricting the actual weight updates to a low-rank subspace. Solves a
  problem this project doesn't currently have: it's LoRA-only today, and
  LoRA's optimizer state is already small (proportional to the tiny
  adapter parameter count, not the frozen base) -- the right answer for a
  hypothetical future *full-parameter* fine-tuning mode, not for
  improving on already-cheap LoRA optimizer state.
- **8-bit block-quantized optimizer moments** (Dettmers, Lewis, Shleifer,
  Zettlemoyer, arXiv:2110.02861, 2022 -- what `bitsandbytes`' `Adam8bit`
  implements). `Algorithm.init_state()`'s existing contract already
  returns "a plain dict of named tensors" without mandating fp32 --
  nothing structurally blocks a quantized-state `Algorithm` variant, but
  CAME and Adafactor were already chosen specifically as memory-frugal
  factored optimizers for a LoRA-sized parameter count, so the marginal
  win from further quantizing an already-small state is real but smaller
  than where the actual VRAM mass is (the frozen base, 3.3).
- **A second event bus for `OffloadOrchestrator`.** Reused `MonitorBus`'s
  existing shape (5.2) instead of inventing a parallel one -- two pub/sub
  systems in one codebase for two similar-but-different purposes would be
  duplication, not design.
- **Redesigning `Node`/`Port`/`ExecutionContext`.** Section 1.1 arrived at
  essentially the same shape independently; section 9.1 confirms it.
  Proposing changes to something already correct, just to have proposed
  something, would be the opposite of the "good code is the only metric"
  standard this design is held to.

---

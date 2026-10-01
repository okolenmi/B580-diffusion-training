*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 4 -- `ResourcePreset` abstraction

**Real memory/objects management inside, per the explicit follow-up
ask -- and a real, reassuring answer to "not sure we have a good base
for it": there already was one, already real and already
production-used, just not yet connected here.**
`nodes/memory/coordinator.py`'s `ResourceCoordinator` is the same
thing `nodes/train/supervised.py`'s `SupervisedLoRATrainerNode` already
registers its own `model`/`optimizer`/`text_encoder` against in a real,
shipping node. `LoRATrainingSkeleton` is now a real `DeviceResident`
itself (`nodes/memory/handle.py`) -- registers its own `.unet`/`.clip`
against an internal coordinator at construction time, so
`footprint_bytes()`/`offload()`/`reload()`/`release()` delegate to
real, already-tested machinery (each already implements
`DeviceResident` itself) instead of a second, hand-rolled, parallel
implementation -- replaced the hand-summed `footprint_bytes()` from the
first pass of this phase. `vae_sd` (real tensor state, not yet wrapped
in any object) is moved/dropped by hand
alongside the coordinator's own work in all three lifecycle methods,
not silently left out of them just because there's no resident object
to register it with yet.

**Not yet built:** validators (per-input human-readable detection text
-- both the checkpoint and LoRA inspection functions this needs now
exist), the parameter-value dictionary (dtype choices), and
list-of-inputs -- the actual `ResourcePreset`/`NodePreset`-satisfying
interface pieces that make this usable *as a node*. Those, plus wiring
this whole thing into an actual `Node` subclass, are Phase 5.

**A working-approach note, not a technical one, worth recording
because it changes how the rest of this redesign should be built:**
earlier framing in this document leaned toward declaring an input only
once its full implementation was ready, to avoid a `Node` contract that
advertises something not yet functional. That instinct is wrong for
this project specifically -- the redesign's whole point is a complete,
correct foundation, even where some of it goes temporarily unused, and
retrofitting a contract after the fact costs more than building it
right the first time. Phase 5's `NodePreset`/`ResourcePreset`
interface should declare the full intended shape of the Resources
Controller now, not grow it incrementally as each piece happens to be
implemented -- and each independent unit (this merge function, the
inspection functions, the construction classes) should be designed
from its own inputs and outputs first, not from how it currently fits
into what already exists; existing code changes to match a better
design when the two conflict, not the other way around.

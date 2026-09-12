*[← docs/design index](README.md)*

# Design goals and constraints

Stated once here, referenced rather than repeated throughout:

1. **VRAM first, speed second -- but not speed-blind.** Every VRAM-saving
   choice below either has no speed cost (pure refactor) or a named,
   estimable one (e.g. activation checkpointing's real recompute cost).
   Nothing here trades speed for VRAM silently or trades VRAM for a
   *bigger* speed cost than the current codebase already accepts,
   without saying so.
2. **Strict OOP.** Behavior lives on objects that implement declared
   interfaces (ABCs). No dict-of-flags threaded through a function, no
   `hasattr()`/`isinstance()` sniffing to decide behavior, no module-level
   mutable state reached for by import.
3. **No singletons.** A component that needs configuration, a device, a
   shared resource pool, or another component gets it through its
   constructor or a method argument -- never by importing a module and
   reading/writing its globals. This is the direct answer to "separation
   from old singleton code": `paths.py`'s `_comfy_dir_override` pattern and
   `core/noise_schedule.py`'s `ALPHA_T`/`SIGMA_T` module-level tensors are
   the two concrete existing instances of what this rule forbids going
   forward (see section 9).
4. **Composition over inheritance, and over rewriting.** Every new
   capability should be addable by writing a new small class that
   implements an existing interface, not by editing a big existing method.
   Where old, verified math is genuinely correct (UNet forward, LoRA
   layers), wrap it -- don't re-derive it -- exactly per the project's
   existing rule.
5. **One reviewed place for device-memory lifecycle.** Every reusable
   device buffer goes through a memory manager. This already exists
   (`nodes/memory/manager.py`'s `MemoryManager`) and is *correct* as a
   low-level primitive -- the design below reuses it unchanged and asks
   "what needs to start using it that doesn't yet," not "how should this
   be redesigned."
6. **Don't overcomplicate.** Every new abstraction below exists because a
   concrete, named problem needs it -- not because it's generically good
   practice. Section 7 lists what was considered and deliberately left
   out, with the reasoning.
7. **Modern techniques earn a place only with real evidence.** Every
   technique adopted below is backed by a specific, checked source (paper,
   arXiv ID) -- not a hunch or something recalled and trusted from memory.
   Each one also gets an honest calibration: adopt now, adopt with a
   stated caveat, or seam-only (room left, not built). A design that
   recommends every paper it read isn't more useful than one that reads
   none of them.

---

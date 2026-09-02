"""Single source of truth for the `t_mode` values RenoiseBatchSourceNode's
and ManagedDatasetSourceNode's own `t_mode` Ports accept.

Deliberately NOT imported from core.noise_schedule, where
sample_timestep() actually implements these five distributions: core/
has an __init__.py that eagerly imports core.unet_wrapper (ComfyUI-
dependent) and core.optimizers, so importing anything under core.* at
module load time would require ComfyUI installed just to build the node
registry / list nodes in the editor -- exactly what
nodegraph_introspect.py's own module docstring promises never happens
("ZERO side effects and ZERO coupling to the rest of the codebase").
renoise.py's own _renoise() already defers its `from core.noise_schedule
import ...` to call time for the same reason; a Port's `choices` is
needed at class-definition time, i.e. module load, so that option isn't
available here.

A deliberate, documented, independent copy of core.noise_schedule's own
list, not an accidental one -- mirrors why
nodes/optimizer/strategy_registry.py centralizes STRATEGIES (one place,
not a hand-duplicated doc string per Port), just without the cross-
package import this particular case can't afford.
"""

T_MODES = ("uniform", "low", "mid", "high", "logit")

"""Single source of truth for the `t_mode` values ManagedDatasetSourceNode's
`t_mode` Port accepts.

Deliberately NOT imported from either side that actually implements or
interprets the modes:

 * core.noise_schedule (where the five static distributions are
   implemented by sample_timestep()) -- core/ has an __init__.py that
   eagerly imports core.unet_wrapper (ComfyUI-dependent) and
   core.optimizers, so importing anything under core.* at module load
   time would require ComfyUI installed just to build the node registry
   / list nodes in the editor -- exactly what
   nodegraph_introspect.py's own module docstring promises never happens
   ("ZERO side effects and ZERO coupling to the rest of the codebase").
 * manager/t_sampling.py (where the train-time modes "adaptive" and
   "exact" are interpreted and validated for draw-time t selection) --
   it imports core (and through it torch), and a Port's `choices` is
   needed at class-definition time, i.e. module load, so that option
   isn't available here either. nodes/dataset/managed.py defers its
   `from manager.loader import ...` into build() for the same reason.

A deliberate, documented, independent copy -- not an accidental one --
mirrors why nodes/optimizer/strategy_registry.py centralizes STRATEGIES
(one place, not a hand-duplicated doc string per Port), just without the
cross-package import this particular case can't afford.
"""

# The five static distributions core.noise_schedule.sample_timestep
# implements, copied faithfully from its T_MODES.
T_MODES = ("uniform", "low", "mid", "high", "logit")

# The five static distributions plus the two train-time modes
# manager/t_sampling.py interprets for single-latent ("lora_raw")
# datasets, where t is chosen at draw time rather than read from the
# shard:
#
#   "adaptive" -- a wired BucketBalance's live per-bucket difficulty
#                 (nodes/train/bucket_balance.py's data side);
#   "exact"    -- t pinned to the t_values list ("500" or
#                 "200,500,800", cycled one value per draw).
#
# Neither is appended to T_MODES itself: T_MODES must stay a faithful
# copy of core's list, and sample_timestep's alpha_beta.get(mode, ...)
# would silently degrade any mode it doesn't know to Beta(1,1) ==
# uniform -- a silently-uniform "adaptive"/"exact" would be a lie. The
# two copies (this constant, t_sampling.py's T_MODES + TRAIN_TIME_MODES)
# are checked against each other by
# nodes/smoke_tests/smoke_test_t_sampling.py.
T_MODES_TRAIN_TIME = (*T_MODES, "adaptive", "exact")

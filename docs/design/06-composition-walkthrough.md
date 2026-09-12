*[← docs/design index](README.md)*

# 6. Composition walkthrough: one LoRA run, under this design

Concrete, to make sections 1-5 legible as a whole rather than a list of
classes. **Most of the classes below are real now** (see each section for
the exact file); `PipelineFactory` and `build_trainable_model`/
`build_optimizer`/`build_text_encoder` (illustrative Builder functions,
standing in for whatever a real script or `ComfyUNetLoRANode.build()`-
style Node actually does) are not, so this still isn't code that runs
exactly as shown -- annotated below for which pieces exist today vs.
which don't. `ManualResourcePolicy`'s constructor below uses its real,
current 3-argument shape (checkpointing, lora_scaling_policy,
parameter_group_policy) -- not the 7-argument version this section's
own text used to illustrate before it was actually built (2.2 covers
why the scope narrowed and how each dropped choice is still made,
just not through this object):

```python
layout = ProjectLayout(...)                              # 1.6, real
device_ctx = DeviceContext.for_device("xpu")              # 1.5, real

schedule = RescaledZeroTerminalSNRSchedule()               # 1.4, real class,
process = DiffusionProcess(schedule, VPredParameterization(), KarrasInputScaler())
# real class, but this specific combination is unvalidated -- see 1.4
# DiffusionProcess.__post_init__ rejects EpsParameterization here -- see 1.4

policy = ManualResourcePolicy(                            # 2.2, real,
    checkpointing=FrozenParamSafeCheckpointing(),         # current 3-argument shape
    lora_scaling_policy=RankStabilizedScaling(),          # 3.2, real
    parameter_group_policy=UniformGroups(),               # 3.4, real -- LoRAPlusGroups(...) real but unvalidated
)
# adapter_strategy, frozen_weight_store, optimizer_execution_strategy, and
# text_encoder_cache are each still their own independent choice, not
# routed through `policy` -- see 2.2 for why each one is scoped out
adapter_strategy = PlainLoRAAdapter()          # 3.1, real -- DoRAAdapter() also real now,
                                                # live-wired via adapter_strategy_scope
frozen_weight_store = BF16WeightStore          # 3.3, real -- NF4WeightStore also real now,
                                                # not yet wired into a forward path
memory = MemoryManager()                                  # 1.3, unchanged, real
coordinator = ResourceCoordinator()                        # 5.1, real

model = build_trainable_model(weights, policy, adapter_strategy,
                               frozen_weight_store, device_ctx)   # a Builder; wires
                                                                   # FrozenWeightStore +
                                                                   # AdapterStrategy + scaling
coordinator.register("model", model)                         # 1.2 DeviceResident, real
optimizer = build_optimizer(model.trainable_parameters(), policy, memory,
                             group_policy=policy.parameter_group_policy())
coordinator.register("optimizer", optimizer)
text_encoder = CachingTextEncoder(build_text_encoder(weights))
coordinator.register("text_encoder", text_encoder)

pipeline = TrainingStepPipeline([                          # 2.1, real
    FetchBatchPhase(prefetching_source),
    EncodeConditioningPhase(text_encoder),
    ForwardPhase(process),
    LossPhase(P2LossWeighting()),                           # 4, real
    BackwardPhase(),
    OptimizerStepPhase(optimizer),
    MonitoringPhase(monitor_handle),
])

orchestrator = OffloadOrchestrator(coordinator, device_ctx)  # 5.2, real
orchestrator.on(CacheRebuildStarting, lambda e, c, d: c.offload_all_except({"model"}))

for step in range(total_steps):
    state = StepState(step=step, batch=None, model=model, device=device)
    state = pipeline.run_step(state)
```

Every object above is independently constructible and independently
testable; nothing is reached for by import; every device-memory owner is
a `DeviceResident` the coordinator actually knows about -- true of the
real pieces today, and the standard the remaining ones (section 10) are
held to as they land.

---

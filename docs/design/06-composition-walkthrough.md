*[← docs/design index](README.md)*

# 6. Composition walkthrough: one LoRA run, under this design

Concrete, to make sections 1-5 legible as a whole rather than a list of
classes. **Most of the classes below are real now** (see each section for
the exact file); `PipelineFactory` and `build_trainable_model`/
`build_optimizer`/`build_text_encoder` (illustrative Builder functions,
standing in for whatever a real script or `ComfyUNetLoRANode.build()`-
style Node actually does) are not, so this still isn't code that runs
exactly as shown -- annotated below for which pieces exist today vs.
which don't. `checkpointing_strategy`/`scaling_policy`/`group_policy`
below are each passed as their own independent value, not bundled into
one policy object -- an earlier version of this walkthrough used a
`ManualResourcePolicy` wrapper for the first two of these (2.2 covers
the full history: why its scope narrowed to 3 of an originally-sketched
7 choices, and why it was removed again later once it turned out no
real Node could ever actually construct and wire one into the graph).

```python
layout = ProjectLayout(...)                              # 1.6, real
device_ctx = DeviceContext.for_device("xpu")              # 1.5, real

schedule = RescaledZeroTerminalSNRSchedule()               # 1.4, real class,
process = DiffusionProcess(schedule, VPredParameterization(), KarrasInputScaler())
# real class, but this specific combination is unvalidated -- see 1.4
# DiffusionProcess.__post_init__ rejects EpsParameterization here -- see 1.4

checkpointing_strategy = FrozenParamSafeCheckpointing()   # 2.2/2.3, real --
                                                            # ComfyUNetLoRANode's
                                                            # use_checkpoint port
scaling_policy = RankStabilizedScaling()                  # 3.2, real -- its own
                                                            # scaling_policy port
group_policy = UniformGroups()                            # 3.4, real -- LoRAPlusGroups(...)
                                                            # also real but unvalidated;
                                                            # Composed*OptimizerNode's own
                                                            # group_policy port, never
                                                            # routed through anything else
# adapter_strategy, frozen_weight_store, optimizer_execution_strategy, and
# text_encoder_cache are each their own independent choice too -- see 2.2
adapter_strategy = PlainLoRAAdapter()          # 3.1, real -- DoRAAdapter() also real now,
                                                # live-wired via adapter_strategy_scope
frozen_weight_store = BF16WeightStore          # 3.3, real -- NF4WeightStore also real now,
                                                # wired into a real forward path via
                                                # NF4LoRALinear/NF4LoRAConv2d
memory = MemoryManager()                                  # 1.3, unchanged, real
coordinator = ResourceCoordinator()                        # 5.1, real

model = build_trainable_model(weights, checkpointing_strategy, scaling_policy,
                               adapter_strategy, frozen_weight_store, device_ctx)
                                                                   # a Builder; wires
                                                                   # FrozenWeightStore +
                                                                   # AdapterStrategy + scaling
coordinator.register("model", model)                         # 1.2 DeviceResident, real
optimizer = build_optimizer(model.trainable_parameters(), memory,
                             group_policy=group_policy)
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

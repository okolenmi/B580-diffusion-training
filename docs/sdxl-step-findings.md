# The SDXL step, measured in this tree: what the 1.3 s is made of

*Measured 2026-10-21 on the B580, with the trainer's own command line
(`--resolution 512 --batch 4 --rank 16 --batch-area 1.5 --param
adapter.base_precision=int8 --param buckets.max_shapes=4`, the `my_set`
dataset).  Every run below is the same command with one thing changed; the
numbers are the minimum over the steps of one shape (3×100×64 latents), so
the allocator's outliers do not speak.*

## The four cells

| base | eager (no graph) | replayed (graph) | the graph's effect | live memory |
|---|---|---|---|---|
| **bf16** | **1134 ms** | 1324 ms | **−14 % — the graph is slower** | 5546 MB |
| **int8** | 1550 ms | **1452 ms** | +6 % | 3489 MB |

And the int8/bf16 ratio in each mode: **+37 % eager** (the int8 path launches
about twice the kernels, so the eager step is host-bound) and **+10 %
replayed** — the same +10-12 % the kernels' own probe measures on this UNet.

## What it says

**1. The step is device-bound.** The graph's replay is one launch, so its
1324 ms is pure device time; the eager step's wall is 1134 ms. The host is
not the bottleneck here — this differs from the old tree, where the step had
enough host work for the graph to be worth 2×.

**2. The graph costs 17 % of the device time.** A replay (1324 ms) is *slower*
than the same work executed eagerly (1134 ms). The mechanism to look at is the
graph's private pool: the graphed run holds ~2 GB more than the eager one
(9299 MB against 8705 MB in the bf16 run; the int8 run's peak is 1.9 GB
higher with the graph too), and the pool's memory layout costs the device
time. The int8 base *reduces* that penalty (its 2 GB of headroom is why its
graph penalty is only 6 %), which is the same effect seen from the other
side.

**3. The int8 base is a memory trade, and it needs the graph to be cheap.**
Eagerly it is +37 % (the extra launches); under the capture +10 %; and it
frees 2.06 GB of live memory (5546 → 3489 MB). Without the graph the int8
base is the slow option; with it, it is the small one.

**4. `buckets.max_shapes` is the difference between a graph and no graph.**
Without it every step's shape is new (4×84×64, 3×64×80, 4×64×80, 4×64×64,
4×88×64, 3×100×64, 4×96×64, 3×116×64 …), the graph's three-sighting warm-up
never completes, each new shape costs 4-5 s to compile and capture, and **the
graph is captured and never replayed** (the step line says `graphs 0 (0
replays)`).

## What to do

* If the step is device-bound, the graph's pool is a net cost: **measure
  without it** (the bf16 eager step is the fastest thing here, 1134 ms) unless
  the host grows again.
* If the int8 base is wanted for the 2 GB, it must be measured *with* the
  graph (1550 → 1452 ms) or it costs 37 %.
* The next lever is the device side, not the host: the GEMMs are 45 % of the
  step's device time, the convolutions 22 % (334.8 M of the UNet's
  parameters, and ConvRot cannot reach a conv), the attention 5 %.

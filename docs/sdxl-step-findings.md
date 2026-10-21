# The SDXL step, measured in this tree: what the 1.3 s is made of

*Measured 2026-10-21 on the B580, with the trainer's own command line
(`--resolution 512 --batch 4 --rank 16 --batch-area 1.5 --param
adapter.base_precision=int8 --graph`, the `my_set` dataset).  Every run below
is the same command with one thing changed; the logs are the trainer's own
step lines.*

## The runs

| run | what changed | the replayed step | the live memory |
|---|---|---|---|
| A | int8 base, **no** `buckets.max_shapes` | **never replays** (0 replays in 20 steps) | 3489 MB |
| B | int8 base, `--pack 0` | never replays (0 in 20) | 3489 MB |
| C | int8 base, `buckets.max_shapes=4` | 1452 ms (12 replays in 24) | 3489 MB |
| D | **bf16** base, `buckets.max_shapes=4` | 1324 ms (12 replays in 24) | 5548 MB |
| E | int8 base, `int8_grad=int8`, `max_shapes=4` | 1481 ms (12 replays in 24) | 3489 MB |

(One shape, 3×100×64 latents, the minimum over its steps; the spread within a
shape is 5-30 %, the outliers are the allocator, not the model.)

## What it says

**1. The step is device-bound, not host-bound.** The graph buys **1.16×** in
this configuration (1539 ms eager → 1324 ms replayed, bf16), not the 2× the
old tree's numbers show. The host is ~15 % of the step.

**2. The int8 base costs +10-12 % and saves 2.06 GB.** 1324 ms → 1452 ms, and
5548 MB → 3489 MB of live memory. That +10-12 % is exactly what the kernels'
own probe measures on this trainer's UNet (210 ms against 190 ms), so the
int8 base is doing what the library says it does.

**3. `buckets.max_shapes` is the difference between a graph and no graph.**
Without it every step's shape is new (4×84×64, 3×64×80, 4×64×80, 4×64×64,
4×88×64, 3×100×64, 4×96×64, 3×116×64 …), the graph's three-sighting warm-up
never completes, each new shape costs 4-5 s to compile and capture, and **the
graph is captured and never replayed**. With `max_shapes=4`: 12 of 24 steps
replay.

**4. The "dense" gradient path is the right default.** `int8_grad=int8`
avoids the dense weight (no copy, so the graph's pool stays small) but its
backward is 1.33× the dense one — the library's number, and it shows up as the
higher tail of the int8 steps.

## What to do

* `--param buckets.max_shapes=4` (or 8) in the command line: without it the
  graph is dead weight and every new shape costs seconds.
* The int8 base is worth having for the **2 GB**, at 10-12 % of the step —
  and only with the graph on (eagerly it is 2.5-2.7× slower: it launches
  about twice the kernels).
* The next lever is the device side, not the host: the GEMMs are 45 % of the
  step's device time, the convolutions 22 % (334.8 M of the UNet's parameters,
  and ConvRot cannot reach a conv), the attention 5 %.

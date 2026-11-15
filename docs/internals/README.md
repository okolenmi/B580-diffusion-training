# Internals

How the harder parts work. Each page describes the code as it is; open questions and measurements point to `../experiment-results.md`.

| Page | Covers |
|---|---|
| [`memory.md`](memory.md) | budget, residency, levers, the ledger, admission |
| [`addons.md`](addons.md) | optional parts of a node, event scopes |
| [`parallel-runs.md`](parallel-runs.md) | modes, planned runs, why weights are not shared |
| [`kernels.md`](kernels.md) | int8 and NF4 frozen weights and the `xpu-kernels` dependency |
| [`capturable-attention.md`](capturable-attention.md) | attention under graph capture |
| [`perf-lab.md`](perf-lab.md), [`acceptance-lab.md`](acceptance-lab.md) | the two measurement tools (`python -m perf`, `python -m accept`) |
| [`anima-model-facts.md`](anima-model-facts.md) | what is known about the Anima files |
| [`clip-vae-schedules.md`](clip-vae-schedules.md) | what hardware taught about CLIP, VAE and schedules |

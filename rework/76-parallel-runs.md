# 76. Parallel runs, instances, and weights shared between processes

Status: **design agreed in direction (owner, 2026-11-11). Built and CPU-tested: modes, planned runs, the scheduler, per-run streams, the Run mode switch and a stop per run. Not built: the instance view, the `WeightSource` seam, the broker (waits for the hardware gates).** Parallel
runs are wanted for stronger cards and for a later in-app image generator that should not push the user to ComfyUI; on a 12 GB card they only make sense if the
big read-only parts (frozen model, VAE, encoders) sit on the device once. The owner chose the hard route first: share the weights, and fall back to full
separation per weight set if the card or driver does not allow it.

## Decisions
* Many runs may be alive at once. The single-active rule of `StartGraphExecution` goes; **memory admission is the only gate** (the ledger already counts every
  live run's claim).
* **Every start creates an instance**: the frozen graph (with its views and layout) and a run record. An instance is chosen from a list, its canvas is read-only,
  it has its own Monitor and statuses `planned`, `running`, `done`, `stopped`, `failed`. "Copy to editor" turns it back into an editable graph.
* **The Run button has a mode**, remembered per browser: *Queue* (start when nothing else runs), *Parallel* (start now or refuse with the memory breakdown),
  *Auto* (parallel if admission passes, otherwise planned, started by the scheduler when a run ends). A planned instance can be cancelled.
* Sharing is an optimisation under a stable seam (`WeightSource`): the same graph runs with `PrivateWeights` (what exists today: each process loads its own)
  or `SharedWeights`. Nothing in a node knows which one it got.

## Instances and the scheduler (no sharing needed)
| Piece | Job |
|---|---|
| `GraphExecution` row | gains `planned` status and `mode`; the frozen graph is the snapshot already stored beside the row |
| `StartGraphExecution` | validates, then either admits and launches, or (mode `queue`/`auto` and no room) stores a `planned` row holding no memory |
| `StartGraphExecution.promote_planned`, `PlannedRunsPump` | every 3 s the pump takes planned rows oldest first and re-runs admission (the demand is looked up again, the budget settings are the ones planned with); the head of the line is never overtaken |
| Stop | planned: the row becomes `stopped` (no claim to release); running: the existing stop |
| Runs panel | lists all instances with status, the selected one is what the canvas and Monitor show (read-only for any started one), `Run` mode switch |
Stop and persisted curves were already per execution id; live streams now are too (`run:<id>:<stream>`, the bus keeps their history until the server restarts). What changed in the editor is the assumption that "the run" is a single thing (`runs.current()`, the Views
`running` flag, the Stop button).

## What can be shared
Frozen base weights (denoiser, VAE, text encoders on the device), each as a **weight set** keyed by content (path, size, mtime, dtype, quantisation). Never
shared: adapters, optimizer state, gradients, activations, caches. A LoRA run applies its adapter as a separate branch, so the base is only read: gate G2
proves that on the card (checksum of the set before and after another process trains on it).

## The broker (design, built after the gates)
A small service owned by the backend (the *weight holder*, its own process so a run's end does not drop the weights) keeps a table of weight sets: key,
footprint, state (`loading`, `resident`, `parked` (in pinned host memory), `evicting`), lessees. Runs never load a shared set themselves; they ask.
* **Lease.** `acquire(key)` returns a handle to the resident tensors and registers the lessee; `release` at the end of the run; a lessee that dies (pid
  checked) loses its lease. A set with no lessees stays resident until memory is wanted (LRU), so the next run starts at once.
* **Park request.** The holder never takes a set from a lessee in the middle of a step. A park request (memory is needed by someone) is sent to the lessees, each
  answers at its **safe point** (between steps, nothing of the set in flight), drops its views and waits; when every lessee has answered the set is moved to
  host memory and the memory is given to whoever asked.
* **Arrival.** A lessee that finds its set parked (or asks for one that is) sends `need(key)` and blocks on the answer: *resident again* (someone else's
  reload, or the holder's own once room exists) or *no room within the timeout* -> the run stops with the named reason `weights_unavailable` and the ledger,
  instead of hanging. Waiting time is reported in the run's step phases as `weights_wait`.
* **No deadlock.** A run takes all the sets it needs in one request ordered by key (all or nothing); a request that cannot be satisfied while holding nothing
  cannot block another that holds something. Priorities: a run that already trains outranks one that is loading; ties by start order.
* **Admission** counts a shared set once, with its lessees listed in the ledger; a new lessee of a resident set costs its private claim only.
* **Per-process private parking stays** (`ResidencyManager` levers for private claims, optimizer state, adapters); only shared sets go through the broker.

## Gates for the hardware (`python -m next.perf ipc`, TESTING item 81)
| Gate | Question | If it fails |
|---|---|---|
| inventory | Which sharing routes does this torch/driver offer (torch IPC for XPU tensors, Level Zero loader, DLPack device kind)? | tells which route to try |
| G1 | Can a device tensor be handed to another process with plain `torch.multiprocessing` (as CUDA does)? | next routes: Level Zero IPC handle (`zeMemGetIpcHandle`/`zeMemOpenIpcHandle`) or dma-buf passing, wrapped by DLPack; native glue, uncertain |
| G2 | Does arithmetic on the borrowed tensor give right results? | route unusable |
| G3 | If the owner frees while a borrower holds it: still valid data, or an error? Free memory before and after the release | tells whether the holder may be one of the lessees or must be separate |
| G4 | Two processes on one card: does time-slicing lose throughput (sum of rates against one alone)? | parallel runs then mean "fit both", not "go faster" |
| G5 | Cost of park and reload of 1 GB, pinned and pageable | sets the wait a lessee must expect and the timeout |
If G1 fails and no other route works, parallel runs fall back to `PrivateWeights` for every set: the scheduler, instances and admission stay as built, and a
second run is admitted only when its whole footprint fits beside the first. One process running several runs as threads is not offered: the step is launch
bound (rework/74), so threads under one interpreter lock would slow every run.

## Order of work
1. Scheduler, per-run streams, Run mode (**built**); then the instance view: read-only canvas from the run's snapshot, own Monitor, "Copy to editor", names, views kept in the snapshot (**next**).
2. The `WeightSource` seam in the load nodes with `PrivateWeights` as the only implementation (behaviour unchanged).
3. Owner runs the gates. Pass -> broker and `SharedWeights`; fail -> stop here and say so in this document.

# 16. The node completion protocol

**Built.** What a node's output *means* — finished, or cut short — travels
beside the output, and a user's Stop now keeps what ran instead of
discarding it.

## The problem

Stop used to end the run at the next node boundary: the runtime checked the
cancel flag before each node and stopped the loop. The trainer exited its
step loop cleanly, but nothing after it ran, so the LoRA saver never
executed and 5,000 steps of progress were lost with the process.

There was also no way for a node to say "this output is partial". Outputs
are plain objects wired through ports (`backend/infrastructure/graph/runtime.py`);
nothing carried meaning *about* them.

## The protocol

`nodes/core.py`:

* `Completion` — frozen: `state` (`"complete"` | `"unfinished"`), and for an
  unfinished one `step`, `total`, `reason`. `COMPLETE` is the default.
  `Completion.unfinished_at(step=, total=, reason="stopped")`.
* A node **reports** with `self.report_completion(...)` from `build()`.
  Nothing reported means complete.
* A node **reads** what each input carried with
  `self.upstream_completion("<input port>")`.

It is a side channel, deliberately **not** a port: ports are typed values
that the graph editor draws and checks, and a flag riding on every wire
would have changed every node's contract. The runtime keeps one
`Completion` per node and hands each node a scoped `ExecutionContext`
(same bus, cancel event and memory by reference; its own completion state).

### Rules (`runtime.py`)

1. **After a stop, only opted-in nodes run.** A node opts in with
   `ACCEPTS_UNFINISHED = True` (default `False`). Everything else is skipped,
   as before.
2. **An opted-in node runs only if some input is unfinished.** A stop means no
   more work; a saver fed purely by finished results has nothing partial to
   handle and is skipped like any other node.
3. **A node with an unfinished input and nothing of its own to say inherits
   it** (the first unfinished input, in port order) — so a pass-through that
   opts in keeps the truth flowing to the saver behind it.
4. A node that was skipped produces no output, so anything depending on it
   is skipped too.
5. Without a stop nothing changes: every node sees `COMPLETE`.

The supervisor still decides the final run status from its own
`_stop_requested`, not from these states.

## Who reports and who reads

* `TrainerNode._report_stopped(step, total)` — both trainer routes call it from
  their cancel branch with the number of **completed** optimizer steps.
* `LoRACheckpointSaverNode` — `ACCEPTS_UNFINISHED = True`, and a new
  `save_unfinished` port (bool, default **on**):
  * finished input → saves to `relative_path`, unchanged;
  * unfinished, `save_unfinished` on → saves to `<stem>_step<N><suffix>`
    (`lora.safetensors` → `lora_step5555.safetensors`); never touches the
    plain name, so a stop can never overwrite or masquerade as a finished LoRA;
  * unfinished, `save_unfinished` off → saves nothing, `saved_path == ""`;
  * stopped at step 0 → saves nothing (the adapter is still its zero-effect
    initial self);
  * unfinished without a step → `_unfinished` in the name, not an invented number.

## Verified how

* `backend/tests/test_graph_runtime.py` — the five rules, per-port state, and
  validation of `Completion`. Fake nodes, CPU.
* `nodes/smoke_tests/smoke_test_lora_saver_unfinished.py` — real safetensors
  writes in a temp directory: names, the checkbox, step 0, no overwrite.
* `nodes/smoke_tests/smoke_test_trainer_cancellation.py`,
  `smoke_test_managed_trainer.py` — both trainers report `unfinished` with the
  completed step count when cancelled.

**Not verified:** a Stop in a real child process on hardware. See the stop
grace below.

## Known limit

The supervisor gives the child 15 s after a stop before SIGKILL. A slow
last step plus a large save could exceed that; the unfinished save would then
be lost. Not changed here — measure a real stop first.

## Using it from a new node

Report partial results (`report_completion(Completion.unfinished_at(...))`);
to handle them, set `ACCEPTS_UNFINISHED = True` and read
`upstream_completion(port)`. A node that does neither is unaffected.

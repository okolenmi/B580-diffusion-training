# Round-4 review material

The external review this repository's `docs/status/fixes-r4.md` works
through, kept for provenance rather than for use.

* `material/TASK-round-4-fixes.md` -- the reviewer's own task breakdown:
  three findings (R4-01..R4-03), plus a list of round-3 leftovers it
  considered still open.
* `material/repro-scripts-round4.tar.gz` -- reproduction scripts. The two
  that exist are extracted to `scripts/repro/` as
  `r17_readiness_fanout.py` and `r18_sweep_deletes_crash_log.py`. `r17` was
  rewritten rather than just extracted: as shipped it built a bare
  `TorchDeviceProbe`, which is precisely the uncached object the fix
  replaced, so it kept reproducing the old behaviour after the fix. See
  "Two corrections to the record" in `docs/status/fixes-r4.md`.
* `material/README-round4.txt` -- the reviewer's index of the same.

## What was reproduced, and what was not

Verified against this tree rather than taken on trust. The review was
written against `82738e2`; this repository has moved a long way since, and
R4-01 in particular had become *worse* rather than better, because
`/installer/devices` was added and it asks the same question of the same
subprocess.

| Finding | Outcome here |
|---|---|
| R4-01 readiness fans out into torch processes | **Real, and reproduced exactly.** 8 concurrent `check.execute()` produced 8 probe invocations with peak concurrency 8; 3 sequential calls produced 3 probes, so there was no cache at all. Each is a 2.2 s torch import that initialises the accelerator runtime. Now 1 and 1. |
| R4-02 the sweep deletes a crashed run's log | **Real.** Reproduced by the reviewer's own `r18` script, which now prints the fixed behaviour: the log survives. |
| R4-03 hermeticity is opt-in | **Real, and reproduced on a bare checkout.** `git archive HEAD` with no `COMFY_DIR`: `test_config` and `test_settings` raised "Cannot find ComfyUI directory" and `test_installer` failed 5 of 64. All three pass now. |

All three reproduced. Nothing in this round was stale.

## One thing the review did not know

R4-01's severity depended on a fact the review had to assume: that a plain
GET is reachable without the Origin check. That is true here — ADR 0001
guards only state-changing methods — and it is why the review was right
that an `<img src=...>` on any open page could trigger it.

What the review could not have known is that this repository had just
gained a second endpoint asking the same question. `GET /installer/devices`
was added for the wizard's GPU choice, and it runs the same subprocess. A
cache placed on `CheckRequirements` — the obvious place — would have left
the newer endpoint exactly as exposed as the one being fixed. The cache is a
wrapper on the *port* for that reason.
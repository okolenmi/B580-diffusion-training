# Round-2 review material

The external review this repository's `docs/status/fixes-r2.md` works
through, kept for provenance rather than for use.

* `material/TASK-backend-fixes.md` -- the work packages (WP-01..WP-22)
  and the findings (N-*, Q*). The tracker cites finding IDs from it, so
  the source of "N-07" is here.
* `material/repro-scripts-round2.tar.gz` -- reproduction scripts for
  round 2. `scripts/repro/` holds r1..r8 extracted from it; **r9..r13
  were only ever in the tarball** and were not extracted, because the
  `core/` removal invalidated three of them and `scripts/repro/` is for
  scripts that run:

  | script | state after 2026-10-02 |
  |---|---|
  | `r9_upload_blocking.py` | would run (N-02 upload; only needs `runs_dir`, still a parameter) |
  | `r10_finished_while_down.py` | **broken** -- uses `seed_run`, removed with the run route |
  | `r11_dataset_file_endpoint.py` | would run |
  | `r12_verify_fixes.py` | **broken** -- uses `StartTraining`, removed with the run route |
  | `r13_sse_coalescing.py` | **broken** -- coalesces `run_progressed`, which no event type is any more |

  `r9` is also the one that wrote a 600 MB file into ComfyUI's real
  `models/loras` when it was first run here. Read its `build_services`
  call before running it again.

This directory was at the project root as `claude's analyze of project 2
(some items may be outdated)/` until 2026-10-02, when WP-22 -- the last
open work package from it -- had been decided and recorded in the tracker
and the folder had become clutter with nothing reading it.
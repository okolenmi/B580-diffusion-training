# Fix pass — progress tracker

This file previously lived only in a chat sandbox and never made it into the
repo — recreating it here now that I'm working directly against
`github.com/okolenmi/B580-diffusion-training` instead of a flat file dump.
This is the canonical version going forward.

Full history of everything fixed in passes 1-3 (resume/cyclic-training
redesign, config migration bug, O(n²) rebatching, discriminated unions,
progress-writer flush throttling, bare-except fixes, deferred ComfyUI path
resolution, etc.) is preserved in chat — happy to re-paste the full writeup
here if useful, but the two items below are what actually changed *now that
I can see the whole repo* (including `manager/` and the full `server/`
package, neither of which were visible before).

## ✅ Resolved / corrected now that the full repo is visible

1. **Confirmed: training runs as a real subprocess, not in-process.**
   `server/process_manager.py` launches training via
   `subprocess.Popen(["python", "-m", "converter.cli", ...])`, and
   `server/monitor.py` just polls `proc.poll()`/reads the progress file —
   it never imports `converter`/`trainer` directly. This resolves the open
   question from the previous pass about `train_step.py` calling
   `gc.disable()` and spawning a background GC thread as import-time side
   effects: since training is always a separate OS process, this can only
   ever affect the training subprocess, never the server itself. Not a
   real risk — no code change needed, just confirming what was previously
   flagged as "worth checking."

2. **Recalibrated `ProgressWriter`'s flush throttle.** The previous fix
   set `_FLUSH_INTERVAL_SEC = 1.0` without visibility into how the server
   actually consumes the file. Now that I can see `server/monitor.py`:
   it polls the progress file every `0.5s` (`time.sleep(0.5)` in
   `RunMonitor.start()`). Flushing only once per second was needlessly
   adding up to another ~0.5s of UI lag on top of the poll cadence for no
   benefit. Lowered to `_FLUSH_INTERVAL_SEC = 0.4` — comfortably under the
   poll interval, so the writer is never the bottleneck, while still
   avoiding a flush syscall on every single training step. File:
   `converter/progress_writer.py`.

3. **`runinfo.py` is confirmed fully dead code across the *entire* repo**,
   not just the files I could see before. `grep -rln "runinfo" .` across
   the whole clone (including `manager/` and `server/`, which weren't
   visible in the earlier passes) turns up nothing but the definition file
   itself — `write_runinfo`/`read_runinfo` have zero callers anywhere.
   Confirmed the server-side progress protocol is entirely
   `.progress.jsonl`-based (`server/progress_file.py`'s
   `ProgressFileReader`, format matches `converter/progress_writer.py`
   exactly — `phase` field, one JSON object per line). Recommend just
   deleting `converter/runinfo.py` next time you're touching that area;
   I'm not doing it unprompted since deleting a whole file is a bigger
   change than a bug fix, but the dead-code case is airtight now.

## Not yet reviewed

`manager/` (dataset.py, db.py, loader.py, builder.py, storage.py,
preview.py) and most of `server/` (config_ui.py, routes_*.py, control.py,
service.py, schemas.py, sse.py, db.py, options.py) haven't had a real bug
pass yet — everything up to this point only covered `converter/` and the
two root entry-point scripts. Worth a dedicated pass once the naming/reorg
question is settled, since renaming touches import paths across all three
packages and it'd be wasteful to bug-fix files whose module paths are about
to change.

## Workflow note

Going forward: I'm working against a clone of the actual GitHub repo,
not a flat set of pasted files. For handing changes back, a `git diff`
patch (plain text, applies with `git apply patch.diff`) is cleaner than
re-pasting whole files or fighting with zip uploads — see the patch
attached alongside this file for the two changes above.

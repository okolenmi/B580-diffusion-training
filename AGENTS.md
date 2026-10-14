# Rules for AI agents working in this repository

Short on purpose. Read it first, every session, after every context compression. The owner has repeated these points
many times; do not make him repeat them again. Longer reasoning is in `rework/90-build-plan.md` (section 0.1 lists
decided questions: do not re-ask them).

## What matters
1. **Do the main task.** Progress on the real work beats polish, side checks and verification theatre. Do not audit
   plan claims against code, re-verify decided things, or build tooling nobody asked for. If something is missing the
   owner will say so.
2. **Do not simplify the goals.** If a goal is hard, build it; if it is impossible here, say what is unverified.
3. **You decide, he reviews.** The owner is less experienced than you in this kind of project and asked you to choose.
   Make the call, record it in one line, move on. Ask only when the answer changes what you build and cannot be
   undone cheaply.
4. **Rewrite, do not port line by line.** Read the old file and its plan, understand what it is for, write the new file.
   Old code stays untouched until the swap. No migration, shims, aliases or compatibility layers (decided).

## Tests: few, and only ones that tell something
- A test earns its place only if it would **fail when the pipeline's structure or contract changes**: graph rules,
  detection results, key translation round trips, plan counts, event order, a loss/target inversion.
- Do **not** test what is not supposed to change, trivial getters, or each line you wrote. No old-vs-new parity suites
  as permanent tests; a one-off comparison script is fine and gets deleted after.
- Sort every check into one of two kinds and say which in the file's docstring:
  **signal** (stays, runs in the final acceptance run) or **temporary** (bring-up only, deleted at the swap).
- Budget: a few checks per module, written after the code works. If a test takes longer to write than the code it
  covers, delete the test. The one big acceptance test comes after the rewrite, not before.
- Cheap whole-tree checks (`scripts/check_imports.py`, `scripts/check_comments.py`, import of every module) are always
  welcome.

## Code and comments
- Comments explain a non-obvious *why*. No history ("moved from"), no measurements, no incident stories in code.
  Measured facts and lessons go in `rework/lessons/` or `docs/known-issues/`.
- Inside `next/` use relative imports only; layer rules are enforced by `scripts/check_imports.py`.
- Keep names honest and short; fix a bad name now, not later.

## Working in the sandbox
- No XPU here: anything hardware-specific is **unverified** until the owner runs it. Say so in the report and the ledger.
- CPU torch works (tiny random models). Sandbox is wiped after 1-2 idle days: deliver work as `git format-patch`
  save points every few steps (verify with `git am` on a clean checkout). Push only if the push check succeeds; at the
  last attempt it was refused (the Claude GitHub App is not installed for this repository).
- At most one sub-agent at a time. Owner limits: about 100k tokens per 5 hours, weekly cap. Do not burn them on tests,
  re-reading, or long reports.

## Communication
- First line of any long task: one sentence saying what you are about to do. At the end: what came out, what is
  unverified, one next step. No recaps of steps, no walls of text.
- Do not ask "should I continue" between phases of an approved plan. Pause only where the plan says to.
- Update `rework/PORTING.md` and the step table in `rework/00-index.md` as files land; they are the resume point.

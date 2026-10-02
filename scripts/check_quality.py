#!/usr/bin/env python3
"""Quality gate: ruff and mypy, measured against a committed baseline.

**The point is not "zero findings". It is that the number cannot go
up.** A tool with no baseline is a tool that gets switched off the first
time it reports something inconvenient; a baseline makes every fix a
visible improvement and every regression a red build.

* a count that *rises* for any rule in any file -> fail, naming the file
  and rule;
* a count that falls -> pass, and say where, so the baseline can be
  re-committed;
* an *unchanged* count with a different distribution (one rule down, one
  up) -> fail, because a per-file/per-rule comparison is what catches
  "fixed a lint by moving it somewhere else".

mypy is optional. It is a dev dependency, not something the app needs to
run, so its absence is a notice rather than a failure -- the same
posture `full_gate.sh` takes for ruff. Its baseline is still recorded,
so installing it later shows the true starting point rather than a
sudden flood.

Usage:
    python scripts/check_quality.py            # check against the baseline
    python scripts/check_quality.py --update   # re-record (deliberate)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "scripts" / "quality_baseline.json"
# Only `backend/`, which is what the review measured and what this
# project owns outright. The other packages are deliberately NOT in the
# baseline: nodes/ and manager/ have their own conventions (lazy imports
# inside functions are load-bearing there, for instance), and core/ is
# being retired -- so their 700-odd findings would sit in the baseline
# forever, which is the same as having no gate at all. Widen this
# deliberately, one package at a time, when a package has an owner for
# driving its own count down.
RUFF_TARGETS = ["backend"]


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def ruff_counts() -> tuple[dict[str, int], bool]:
    """(per "file|rule" -> count, available)."""
    available = subprocess.run(
        [sys.executable, "-c", "import ruff"],
        cwd=ROOT, capture_output=True, check=False,
    ).returncode == 0
    if not available:
        return {}, False

    code, out = _run(
        [sys.executable, "-m", "ruff", "check", *RUFF_TARGETS,
         "--output-format", "json"]
    )
    # ruff exits 1 when it has findings; that is data, not a failure.
    if code not in (0, 1):
        print(out, file=sys.stderr)
        raise SystemExit("ruff could not run")
    try:
        findings = json.loads(out or "[]")
    except json.JSONDecodeError:
        print(out, file=sys.stderr)
        raise SystemExit(
            "ruff produced output this script cannot read"
        ) from None

    counter: Counter[str] = Counter()
    for item in findings:
        key = f"{item['filename'].replace(str(ROOT) + '/', '')}|{item['code']}"
        counter[key] += 1
    return dict(counter), True


def mypy_counts() -> tuple[dict[str, int], bool]:
    available = subprocess.run(
        [sys.executable, "-c", "import mypy"],
        cwd=ROOT, capture_output=True, check=False,
    ).returncode == 0
    if not available:
        return {}, False

    code, out = _run(
        [sys.executable, "-m", "mypy", "--no-error-summary",
         "--no-pretty", "--output", "json", "backend"]
    )
    if code not in (0, 1):
        print(out, file=sys.stderr)
        raise SystemExit("mypy could not run")
    # mypy emits either one JSON array or (with --output json and no
    # summary) one JSON object per line, and can interleave progress
    # text. Accept both rather than pinning to whichever version happens
    # to be installed.
    results: list[dict] = []
    stripped = (out or "").strip()
    if stripped.startswith("["):
        results = json.loads(stripped)
    else:
        for line in stripped.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not results and stripped:
        print(out, file=sys.stderr)
        raise SystemExit("mypy produced output this script cannot read")

    counter: Counter[str] = Counter()
    for item in results:
        if item.get("ignore_in_config") or not item.get("code"):
            continue
        raw = item.get("file") or "?"
        # mypy reports repo-relative paths when run from the root, but be
        # tolerant of an absolute one rather than crashing the gate on a
        # cosmetic difference.
        path = raw
        try:
            path = Path(raw).resolve().relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            path = raw
        counter[f"{path}|{item['code']}"] += 1
    return dict(counter), True


def compare(baseline: dict, current: dict, tools: list[str]) -> tuple[list[str], list[str]]:
    """(regressions, improvements) as human-readable lines."""
    regressions: list[str] = []
    improvements: list[str] = []
    for name in tools:
        before = baseline.get(name, {})
        after = current[name]
        for key in sorted(set(before) | set(after)):
            old_count = before.get(key, 0)
            new_count = after.get(key, 0)
            if new_count > old_count:
                regressions.append(f"{name}: {key}: {old_count} -> {new_count}")
            elif new_count < old_count:
                improvements.append(f"{name}: {key}: {old_count} -> {new_count}")
    return regressions, improvements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_true",
        help="re-record the baseline from the current state (deliberate)",
    )
    args = parser.parse_args()

    ruff, ruff_ok = ruff_counts()
    mypy, mypy_ok = mypy_counts()

    current = {"ruff": ruff, "mypy": mypy}
    tools = [name for name, ok in (("ruff", ruff_ok), ("mypy", mypy_ok)) if ok]
    unavailable = [
        name for name, ok in (("ruff", ruff_ok), ("mypy", mypy_ok)) if not ok
    ]

    if not tools:
        print("check_quality: neither ruff nor mypy is installed; "
              "nothing to check (pip install -r requirements-dev.txt)")
        return 0

    for name in unavailable:
        print(f"check_quality: {name} is not installed; skipped with a notice")

    if args.update:
        BASELINE_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name in tools:
            print(f"check_quality: baseline recorded -- {name}: "
                  f"{len(current[name])} distinct findings, "
                  f"{sum(current[name].values())} total")
        return 0

    if not BASELINE_PATH.exists():
        print(f"check_quality: no baseline at {BASELINE_PATH.relative_to(ROOT)}; "
              f"run with --update to record one")
        return 1

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressions, improvements = compare(baseline, current, tools)

    for line in improvements:
        print(f"  improved  {line}")
    if improvements:
        print("check_quality: re-run with --update to bank the improvement")

    if regressions:
        print("\ncheck_quality: REGRESSED -- findings must never increase:")
        for line in regressions:
            print(f"  {line}")
        print("\nFix them, or --update if raising the baseline is deliberate "
              "and justified in the commit message.")
        return 1

    for name in tools:
        total = sum(current[name].values())
        print(f"check_quality: {name} {total} finding(s), baseline held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Sample the on-disk compiler caches once per second into a CSV (run it in the background
next to a training run; stop it with Ctrl-C or by killing it).

    python3 watch_cache.py --out cache_watch.csv &
Columns: t (unix seconds, same clock as steps.jsonl 'wall'), name, files, bytes.
A cache that grows during slow steps means compiled code is being generated and persisted;
a cache that sits at a constant size while steps stay slow means either nothing is compiled
or the cache is full and evicting (compare its size with the limit: for the Intel driver
cache the limit is NEO_CACHE_MAX_SIZE).
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path


def default_dirs() -> dict[str, str]:
    home, user = Path.home(), os.environ.get("USER", "user")
    cands = {
        "neo": os.environ.get("NEO_CACHE_DIR") or str(home / ".cache" / "neo_compiler_cache"),
        "sycl": os.environ.get("SYCL_CACHE_DIR") or str(home / ".cache" / "libsycl_cache"),
        "inductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR") or f"/tmp/torchinductor_{user}",
        "triton": os.environ.get("TRITON_CACHE_DIR") or str(home / ".triton" / "cache"),
        "sycl_kernels": str(home / ".cache" / "sycl_kernels"),
    }
    return cands


def dir_stats(path: str) -> tuple[int, int]:
    files = size = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        files += 1
                        try: size += e.stat(follow_symlinks=False).st_size
                        except OSError: pass
        except OSError:
            pass
    return files, size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until interrupted")
    ap.add_argument("--dir", action="append", default=[], metavar="NAME=PATH")
    args = ap.parse_args(argv)
    dirs = default_dirs()
    for item in args.dir:
        k, _, v = item.partition("="); dirs[k] = v
    with open(args.out, "w", newline="", buffering=1) as fh:
        w = csv.writer(fh); w.writerow(["t", "name", "files", "bytes", "exists"])
        t_end = time.time() + args.duration if args.duration else None
        try:
            while t_end is None or time.time() < t_end:
                now = round(time.time(), 3)
                for name, path in dirs.items():
                    exists = os.path.isdir(path)
                    f, b = dir_stats(path) if exists else (0, 0)
                    w.writerow([now, name, f, b, int(exists)])
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Known issues / suspicious findings / deferred work

Two rules that matter before editing anything here:

1. **These files are cited from source comments and docstrings.** Several
   `nodes/` comments and design docs point at specific findings by file, so
   deleting a file or an entry breaks references you can't see from here.
2. **`resolved.md`'s hardware measurements are the only record.**
   `runs/` and `datasets/` are gitignored and empty in a fresh clone, so
   the numbers recorded in those entries (VRAM reserved/peak figures,
   steps/sec, OOM thresholds) exist nowhere else. Don't trim them for
   length.

| File | What's in it |
|---|---|
| [`open.md`](open.md) | Confirmed or suspected issues with no fix landed yet, plus one closed-by-measurement finding. Check here first if you've hit something odd. |
| [`resolved.md`](resolved.md) | Worth mentioning resolved cases (don't add new if they don't give valuable information) |

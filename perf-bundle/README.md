# Hardware check bundle

| step | result | seconds |
|---|---|---:|
| hardware | ok | 2 |
| allocator expandable | ok | 2 |
| allocator plain | ok | 2 |
| bench | ok | 24 |
| matrix | ok | 3461 |
| verdict | ok | 2 |

Send `bundle.zip` (text files only), or these files: `hardware/hardware.md`, `bench/bench.md`, `matrix/summary.md`, the `report.md` inside each `matrix/<scenario>/` folder,
`verdict/verdict.md` (the acceptance lab's pass/fail per check) and `logs/` for anything that failed. The allocator steps print one JSON line each (in `logs/`): equal numbers between the two mean the allocator setting changes nothing.

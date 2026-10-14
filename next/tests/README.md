# Test kinds in `next/tests`

**signal** (stays for the acceptance run) and **temporary** (bring-up only, delete at the swap). See `AGENTS.md`.

| file | kind | note |
|---|---|---|
| `check_graph_core.py` | signal | graph rules, events, validation, runner, settings; trim redundant cases when touched |
| `check_detection.py` | signal | file recognition and sandboxing, including refusal cases |
| `check_families_torch.py` | schedules/VAE/CLIP: temporary except the schedule inversion and key-translation round trip (signal) |
| `check_sdxl.py` | temporary except the blend/segments contract (signal) |
| `check_adapters.py` | temporary except the plan counts and header-plan equals injected-plan (signal) |

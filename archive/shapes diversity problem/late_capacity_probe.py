"""Does ONEDNN_PRIMITIVE_CACHE_CAPACITY still work if it is set AFTER the device exists?

This decides whether an adaptive capacity is implementable at all.

`set_xpu_perf_env_vars()` is called in `graph_task_worker.py` *before* the graph
is parsed -- deliberately, because SYCL reads its variables at runtime init and
setting them late is documented there as too late. At that moment nobody knows
which dataset the graph will use, so nobody knows how many distinct latent
shapes it has, so a capacity sized to the dataset cannot be set there.

So the question is whether oneDNN's primitive cache reads this variable lazily
instead. If it does, the trainer can count the dataset's distinct shapes once it
has them and size the cache to the dataset. If it does not, then a correct
capacity has to be known before torch is imported, which means it has to come
from somewhere other than the dataset -- a config value, or a graph-level
statement.

This wrapper deliberately makes the timing as hostile as possible: it imports
torch AND touches the XPU device (forcing runtime init and oneDNN loading) and
only then sets the variable. If the fix still works under those conditions, the
variable is read lazily and the adaptive design is available.

Run as:  python3 late_capacity_probe.py --capacity 2048 <hw_validate args...>
"""

import os
import runpy
import sys
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
sys.path.insert(0, str(REPO))


def main() -> int:
    argv = sys.argv[1:]
    capacity = None
    if argv and argv[0] == "--capacity":
        capacity = argv[1]
        argv = argv[2:]

    # The production variables, at the normal time.
    from nodes.xpu_env import set_xpu_perf_env_vars
    set_xpu_perf_env_vars()

    # Now do the hostile part: import torch and actually touch the device, so
    # the SYCL runtime has initialised and oneDNN is loaded, before the
    # capacity is set.
    import torch
    torch.zeros(8, device="xpu").sum().item()
    free, _total = torch.xpu.mem_get_info()
    print(f"[late-set probe] torch imported and device touched "
          f"(free {free / 2**20:.0f} MB) BEFORE setting the capacity", flush=True)

    if capacity is not None:
        os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = capacity
        print(f"[late-set probe] ONEDNN_PRIMITIVE_CACHE_CAPACITY={capacity} set "
              f"AFTER device init", flush=True)

    # Confirm the variable was not already set, so this is a real late set and
    # not a leftover from the environment.
    print(f"[late-set probe] value in os.environ at trainer start: "
          f"{os.environ.get('ONEDNN_PRIMITIVE_CACHE_CAPACITY')}", flush=True)

    # Run hw_validate in THIS process, via runpy rather than an import.
    #
    # The import would break two things at once: `scripts/` is not a package,
    # so `import scripts.hw_validate` reaches a namespace package that the
    # dependency test cannot tell from a third-party one
    # (backend/tests/test_declared_dependencies.py); and an import would run
    # that module under a different __name__ than its own __main__ guard
    # expects. runpy.run_path executes it as __main__ in the current process,
    # which is what the measurement needs -- torch is already imported and the
    # device already touched in THIS interpreter, so the capacity really is
    # being set after init, and the trainer really does run here.
    sys.argv = ["hw_validate.py"] + argv
    runpy.run_path(str(REPO / "scripts" / "hw_validate.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

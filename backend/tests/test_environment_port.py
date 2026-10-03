"""The environment port: what is installed, and what the card is.

Two ports that fail differently, which is why they are two:
`PackageInventory` is cheap, total and never raises, so "is torch
installed" can be answered without loading torch. `DeviceProbe` imports
torch, so it can fail in ways that are all *data* and returns a result
object rather than raising.

**The check this file exists for is the agreement between the two device
probes.** `report()` asks about `current_device()`; `devices()` enumerates.
They are separate subprocesses answering the same question about the same
card, and when they disagreed the enumerated list reported the B580 on this
machine as *absent* while `report()` reported it present -- because the
list child's success path filled in the name and memory and never set
`ok = True`. Either probe alone would have been reported as a fact about
the machine. Only comparing them shows that neither is.

So the real assertions below run against the real card, and compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.environment import (  # noqa: E402
    IMPORT_NAMES,
    DeviceReport,
    MetadataPackageInventory,
    TorchDeviceProbe,
    _last_json_object,
)
from backend.application.ports.requirements_manifest import REQUIREMENTS  # noqa: E402
from backend.tests.support import (  # noqa: E402
    FakeDeviceProbe,
    check,
    finish,
)


# ==========================================================================
print("-- finding the answer in a noisy stream --")

# The XPU runtime prints a Mesa banner to stdout before answering, on this
# machine, every time. A probe that parsed the whole stream would call a
# working card broken.
banner = (
    "Rusticl warning: Patched Mesa libclc not detected.\n"
    "visit https://gitlab.freedesktop.org/karolherbst/mesa-libclc\n"
    '{"ok": true, "backend": "xpu", "name": "Intel(R) Arc(TM) B580 Graphics"}\n'
)
check(_last_json_object(banner) is not None,
      "a banner before the answer does not hide it")
check(_last_json_object(banner)["name"].startswith("Intel"),
      "and the answer is the right one")

check(_last_json_object("") is None, "empty output has no answer")
check(_last_json_object("not json at all") is None,
      "prose that is not JSON has no answer")
check(_last_json_object('{"name": "x"}') is None,
      'a JSON object with no "ok" is not an answer -- "ok" is the contract')
check(_last_json_object('{"ok": true}\n{"ok": false, "reason": "later"}')["reason"]
      == "later",
      "and when there are two, the last one is the answer")

# ==========================================================================
print("\n-- name maps, which is where a check silently goes wrong --")

inventory = MetadataPackageInventory()

# python-multipart is imported as multipart. A version lookup keyed the
# wrong way reports a package missing on a machine that has it.
check(inventory.import_name_for("python-multipart") == "multipart",
      f"python-multipart imports as multipart "
      f"({inventory.import_name_for('python-multipart')})")
check(inventory.distribution_for("multipart") == "python-multipart",
      f"and back again ({inventory.distribution_for('multipart')})")
check(inventory.import_name_for("torch") == "torch",
      "a package whose names agree maps to itself")

# Round-tripping every mapped name must be lossless, or a lookup by import
# name would find a package under a name pip does not use.
roundtripped = {
    dist: inventory.distribution_for(inventory.import_name_for(dist))
    for dist in IMPORT_NAMES
}
check(all(roundtripped[dist] == dist for dist in IMPORT_NAMES),
      f"every mapped name round-trips ({roundtripped})")

# A name that is not in the map is its own distribution.
check(inventory.distribution_for("nonexistent_module") == "nonexistent_module",
      "an unmapped name is its own distribution")

# Total by contract: a name that is absent is None, never an exception.
check(inventory.version_of("definitely-not-installed-xyzzy") is None,
      "an absent package is None, not a fault")

# And the manifest's import names must come from the same map, or the
# readiness screen and the training code could disagree about what
# "installed" means for the same package.
manifest_names = {r.distribution: r.import_name for r in REQUIREMENTS}
check(all(
    manifest_names[dist] == inventory.import_name_for(dist)
    for dist in manifest_names
), f"the manifest and the inventory agree on import names ({manifest_names})")

# ==========================================================================
print("\n-- the two device probes must agree about the same card --")

# One torch import, in process, reused by every check below. The subprocess
# path is checked separately for agreement with this one.
probe = TorchDeviceProbe(backend="xpu", _in_process=True)
listed, list_reason = probe.devices_with_reason()
reported = probe.report()

check(list_reason is None,
      f"listing devices gives no reason when it worked ({list_reason!r})")
check(len(listed) >= 1,
      f"and at least this machine's card is listed ({len(listed)})")

present_listed = [d for d in listed if d.present]
check(len(present_listed) == len(listed),
      f"every enumerated device is present, or says why it is not "
      f"({[(d.present, d.reason) for d in listed]})")

# The invariant that was missing. `report()` is the current device, so it
# must appear in the enumeration with the same name and memory -- and here
# they did not: the list said present=False for a card the report called
# present, and neither probe was wrong on its own.
check(reported.present,
      f"the current device is present ({reported.reason!r})")
matching = [d for d in listed if d.name == reported.name]
check(bool(matching),
      f"the enumerated list contains the device report() describes "
      f"(reported {reported.name!r}, listed {[d.name for d in listed]})")
check(all(d.present for d in matching),
      f"and agrees it is present ({[d.present for d in matching]})")
check(all(
    abs((d.total_memory_mb or 0) - (reported.total_memory_mb or 0)) < 1
    for d in matching
), f"and agrees how much memory it has ({reported.total_memory_mb} vs "
   f"{[d.total_memory_mb for d in matching]})")

# ==========================================================================
print("\n-- the subprocess path agrees with the in-process one --")

# Both run the same source, but one is exec'd in a captured stdout and the
# other is a real child. A difference between them would mean the probe's
# answer depends on how it was asked.
subprocess_listed, subprocess_reason = TorchDeviceProbe(backend="xpu").devices_with_reason()
check(subprocess_reason is None,
      f"the subprocess lists without complaint ({subprocess_reason!r})")
check([(d.present, d.name, d.total_memory_mb) for d in subprocess_listed]
      == [(d.present, d.name, d.total_memory_mb) for d in listed],
      f"and returns the same rows as the in-process path "
      f"({[(d.name, d.present) for d in subprocess_listed]})")

# ==========================================================================
print("\n-- a backend with nothing behind it --")

# This torch build has torch.cuda and zero cuda devices, which is the case
# a naive "is the attribute there" check gets wrong.
cuda_rows, cuda_reason = TorchDeviceProbe(backend="cuda", _in_process=True).devices_with_reason()
check(isinstance(cuda_rows, tuple),
      f"a backend with no devices returns a tuple, not a fault "
      f"({cuda_rows!r})")
check(cuda_reason is None,
      f"and no reason, because the check *ran* and found none "
      f"({cuda_reason!r})")
check(cuda_rows == () or all(isinstance(d, DeviceReport) for d in cuda_rows),
      f"either nothing or DeviceReports ({cuda_rows!r})")

# And `report()` must not claim a card is there just because the attribute
# exists.
cuda_report = TorchDeviceProbe(backend="cuda", _in_process=True).report()
check(not cuda_report.present,
      f"report() agrees there is no cuda device ({cuda_report.reason!r})")

# A backend name torch has never heard of.
unknown_rows, unknown_reason = TorchDeviceProbe(
    backend="nonesuch", _in_process=True
).devices_with_reason()
check(unknown_rows == () and unknown_reason,
      f"a backend torch does not have is a reason, not an exception "
      f"({unknown_reason!r})")

# ==========================================================================
print("\n-- the fallback, and telling it apart from an answer --")

# A port that only knows how to report one device must still answer, and
# must admit it did not enumerate.
fake = FakeDeviceProbe()
check(len(fake.devices()) == 1,
      f"a probe that does not enumerate falls back to the current device "
      f"({len(fake.devices())})")
check(fake.enumerate_all is False,
      "and says so, so a one-card answer is distinguishable from an "
      "unchecked one")
check(TorchDeviceProbe(backend="xpu", _in_process=True).enumerate_all is True,
      "and the real probe reports that it did enumerate")

finish()
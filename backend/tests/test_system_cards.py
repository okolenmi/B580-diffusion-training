"""Listing graphics cards without torch: the setup page asks before any wheel is installed.

Commands and sysfs are scripted, so no card or OS tool is needed. The probe test runs the real child in an interpreter where torch
cannot be imported (blocked), which is exactly the fresh-clone situation.

Run directly: python backend/tests/test_system_cards.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.environment.torch_probe import TorchDeviceProbe
from backend.application.ports.system_cards import SystemCards
from backend.tests.support import check, finish

LSPCI = (
    '00:02.0 "VGA compatible controller" "Intel Corporation" "Raptor Lake-S GT1 [UHD Graphics 770]" -r04 "ASUSTeK" "Device 8882"\n'
    '03:00.0 "VGA compatible controller" "Intel Corporation" "Battlemage G21 [Arc B580]" -r01 "Intel" "Device 1234"\n'
    '00:1f.3 "Audio device" "Intel Corporation" "Raptor Lake High Definition Audio" -r11 "ASUSTeK" "Device 87d5"\n'
)


def test_lspci_rows_are_display_adapters_only() -> None:
    cards = SystemCards("linux", runner=lambda cmd: LSPCI).list()
    check([c.name for c in cards] == ["Intel Raptor Lake-S GT1 [UHD Graphics 770]", "Intel Battlemage G21 [Arc B580]"],
          f"lspci lists the two display adapters and not the audio device ({[c.name for c in cards]})")


def test_sysfs_is_the_fallback_when_lspci_is_missing() -> None:
    def missing(cmd):
        raise FileNotFoundError(cmd[0])

    with tempfile.TemporaryDirectory() as tmp:
        node = Path(tmp) / "card0" / "device"
        node.mkdir(parents=True)
        (node / "vendor").write_text("0x8086\n")
        (node / "device").write_text("0xe20b\n")
        cards = SystemCards("linux", runner=missing, drm_root=Path(tmp)).list()
    check(len(cards) == 1 and "Intel" in cards[0].name and "0xe20b" in cards[0].name, f"sysfs ids name the card ({cards})")


def test_windows_json_one_or_many_rows() -> None:
    one = SystemCards("win32", runner=lambda cmd: '{"Name":"Intel(R) Arc(TM) B580 Graphics","AdapterRAM":4293918720}').list()
    many = SystemCards("win32", runner=lambda cmd: '[{"Name":"A"},{"Name":"B"},{"Name":null}]').list()
    check([c.name for c in one] == ["Intel(R) Arc(TM) B580 Graphics"], "a single object is one card")
    check([c.name for c in many] == ["A", "B"], "a list keeps named cards and skips unnamed ones")


def test_nothing_it_cannot_read_raises() -> None:
    check(SystemCards("linux", runner=lambda cmd: "garbage \"unterminated").list() == (), "unparseable output is no cards")
    check(SystemCards("win32", runner=lambda cmd: "not json").list() == (), "bad JSON is no cards")
    check(SystemCards("darwin").list() == (), "an unsupported system is no cards")


def test_missing_torch_falls_back_to_the_system_list() -> None:
    gone = {"ok": False, "torch_missing": True, "reason": "torch is not importable"}
    probe = TorchDeviceProbe(system_cards=SystemCards("linux", runner=lambda cmd: LSPCI))
    rows, reason = probe._without_torch(gone)
    check(reason is None and len(rows) == 2 and rows[1].present and "after the install" in (rows[1].detail or ""),
          "torch absent: the system's cards come back with a note")
    rows, reason = probe._without_torch({"ok": False, "reason": "this torch build has no 'xpu' backend"})
    check(rows == () and "no 'xpu'" in reason, "a torch that is present but lacks xpu is still reported as that")
    rows, reason = TorchDeviceProbe(system_cards=SystemCards("darwin"))._without_torch(gone)
    check(rows == () and "not importable" in reason, "no system answer: the original reason stands")


if __name__ == "__main__":
    test_lspci_rows_are_display_adapters_only()
    test_sysfs_is_the_fallback_when_lspci_is_missing()
    test_windows_json_one_or_many_rows()
    test_nothing_it_cannot_read_raises()
    test_missing_torch_falls_back_to_the_system_list()
    finish()

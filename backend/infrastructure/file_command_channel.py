"""FileCommandChannel -- the server side of the command inbox: one folder per run next to its event file."""

from __future__ import annotations

from pathlib import Path

from ..application.ports.command_channel import CommandChannel
from .command_inbox import CommandOutbox, commands_dir_for


class FileCommandChannel(CommandChannel):
    def __init__(self, scratch_dir: Path) -> None:
        self._scratch = scratch_dir

    def submit(self, execution_id: int, node: str, action: str) -> str:
        return CommandOutbox(commands_dir_for(self._scratch, execution_id)).submit(node, action)

"""Training state that survives a power cut: what a run needs, written so that a cut at any moment leaves a usable file.

``ResumeStore`` owns the files. A state is two files, ``resume_<step>.safetensors`` (the tensors) and ``resume_<step>.json`` (the
step, the size and the SHA-256 of the tensor file, and what the run was); the json is written last, so a state without one never
finished. The newest ``KEEP`` complete states stay, the older ones go only after a newer state is complete. Loading checks the size
and the hash and falls back to the previous state when the newest is damaged.

``ResumeCheckpoint`` knows what goes into a state (the adapter, the optimizer's moments, the weight average) and puts it back."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass(frozen=True)
class ResumeState:
    step: int
    position: int  # batches the data stream had handed out
    identity: str
    tensors: dict[str, Tensor]
    meta: dict


class ResumeStore:
    KEEP = 2

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _names(self, step: int) -> tuple[Path, Path]:
        stem = self.directory / f"resume_{step:08d}"
        return stem.with_suffix(".safetensors"), stem.with_suffix(".json")

    def save(self, step: int, position: int, identity: str, tensors: dict[str, Tensor], meta: dict) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        data, record = self._names(step)
        scratch = data.with_suffix(".safetensors.tmp")
        save_file(tensors, str(scratch))
        self._flush(scratch)
        digest = self._digest(scratch)
        os.replace(scratch, data)
        body = {"step": step, "position": position, "identity": identity, "bytes": data.stat().st_size, "sha256": digest, "meta": meta}
        note = record.with_suffix(".json.tmp")
        note.write_text(json.dumps(body), encoding="utf8")
        self._flush(note)
        os.replace(note, record)  # the state exists from here on
        self._sync_directory()
        self._prune()
        return data

    def latest(self, say: Callable[[str], None] = lambda message: None) -> ResumeState | None:
        """The newest state that is complete and whole; a damaged one is reported and skipped."""
        for record in sorted(self.directory.glob("resume_*.json"), reverse=True) if self.directory.exists() else []:
            try:
                body = json.loads(record.read_text(encoding="utf8"))
                data = record.with_suffix(".safetensors")
                if data.stat().st_size != body["bytes"] or self._digest(data) != body["sha256"]:
                    raise ValueError("size or hash differs from what was written")
                return ResumeState(int(body["step"]), int(body["position"]), body["identity"], load_file(str(data)), body["meta"])
            except Exception as error:  # noqa: BLE001 -- any damage means: try the one before
                say(f"resume state {record.name} is not usable ({type(error).__name__}: {error}); trying the one before it")
        return None

    def clear(self) -> None:
        for path in self.directory.glob("resume_*") if self.directory.exists() else []:
            path.unlink(missing_ok=True)

    def _prune(self) -> None:
        complete = sorted(p.stem for p in self.directory.glob("resume_*.json"))
        keep = set(complete[-self.KEEP:])
        for path in self.directory.glob("resume_*"):
            stem = path.name.split(".")[0]
            if stem not in keep:
                path.unlink(missing_ok=True)

    @staticmethod
    def _flush(path: Path) -> None:
        with open(path, "rb+") as handle:
            os.fsync(handle.fileno())

    def _sync_directory(self) -> None:
        try:
            fd = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return  # not every platform lets a directory be opened; the files themselves were flushed
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        return digest.hexdigest()


class ResumeCheckpoint:
    """What a state holds and how it goes back: the adapter's factors, the optimizer's moments and counter, the weight average."""

    def __init__(self, adapted, optimizer, ema=None):
        self.adapted, self.optimizer, self.ema = adapted, optimizer, ema

    def capture(self) -> tuple[dict[str, Tensor], dict]:
        tensors: dict[str, Tensor] = {}
        for path, state in self.adapted.state().items():
            for name, param in state.items():
                tensors[f"adapter/{path}/{name}"] = param.detach().to("cpu", copy=True).contiguous()
        for name, tensor in self.optimizer.state_tensors().items():
            tensors[f"optimizer/{name}"] = tensor
        meta = {"optimizer_steps": self.optimizer.algorithm.counter()}
        if self.ema is not None:
            averages, ema_meta = self.ema.state()
            tensors.update({f"ema/{name}": t for name, t in averages.items()})
            meta["ema"] = ema_meta
        return tensors, meta

    def restore(self, state: ResumeState) -> None:
        by_prefix = lambda prefix: {k[len(prefix):]: v for k, v in state.tensors.items() if k.startswith(prefix)}  # noqa: E731
        saved = by_prefix("adapter/")
        lacking = [path for path in self.adapted.layers if f"{path}/lora_A" not in saved or f"{path}/lora_B" not in saved]
        if lacking:
            raise ValueError(f"the saved state has no weights for {len(lacking)} adapter layers (first: {lacking[0]}); it belongs to another adapter")
        for path, layer in self.adapted.layers.items():
            layer.load_adapter_state({name: saved[f"{path}/{name}"] for name in layer.adapter_state() if f"{path}/{name}" in saved})
        self.optimizer.load_state_tensors(by_prefix("optimizer/"))
        self.optimizer.algorithm.set_counter(int(state.meta["optimizer_steps"]))
        if self.ema is not None:
            if "ema" not in state.meta:
                raise ValueError("this run keeps weight averages but the saved state has none")
            self.ema.restore(by_prefix("ema/"), state.meta["ema"])

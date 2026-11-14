"""LoRACheckpointSaverNode: writes a trained-weights state dict to disk.

Depends on TrainedWeightsExportable (nodes/model/handle.py), not a
concrete model class. TrainableModel extends that interface, so a live,
still-training ComfyUNetTrainableModel is one implementer; a
LoRAPhaseSplitNode's completed_generation snapshot (FrozenLoRASnapshot,
nodes/model/lora_phases.py -- not a TrainableModel at all, no
forward/train/eval/to) is another. Both satisfy the same real
issubclass() check the graph editor does on declared port types, so one
saver node genuinely serves both wires rather than needing a second,
near-duplicate node class for the snapshot case.

Writes the dict straight to safetensors itself rather than going through
core.save.save_lora_checkpoint -- that legacy function reads weights via
ComfyUNetWrapper.get_lora_weights(), which lora.py's own isinstance
gate would skip for a phase-split model's LoRAGeneration layers (see
nodes/model/lora_phases.py); ComfyUNetTrainableModel.trained_state_dict()
already resolves that, and there's nothing left for the legacy function
to add on top of an already-detached, already-CPU state dict.

Known gap, not fixed here because nothing in nodes/ can trigger it yet:
core.save.save_lora_checkpoint also temporarily removes a FusedXPUAdafactor's
backward hooks during the read, to avoid a hook firing mid-save. That
race needs a *live, still-training* model and a save happening while
training is paused mid-loop -- still not reachable: the one in-loop save
caller that now exists (ManagedLoRATrainerNode's save_every_n_steps)
calls this module's save_trained_weights() between optimizer steps,
never mid-backward, where hooks are the only thing that could fire (see
that function's own docstring). Still worth revisiting together if a
save ever lands mid-backward -- and this node's previous version, also
never passed an optimizer through here either, so this isn't a
regression.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..components.layout import ProjectLayout
from .handle import TrainedWeightsExportable
from .node import CheckpointSaverNode


def save_trained_weights(model: TrainedWeightsExportable, relative_path: str,
                         layout: ProjectLayout | None = None) -> str:
    """Atomic safetensors write of a TrainedWeightsExportable under the
    sandboxed LoRA directory -- one implementation shared by
    LoRACheckpointSaverNode.build() and ManagedLoRATrainerNode's own
    save_every_n_steps (the in-training cadence port), so the mid-run
    files and the end-of-run graph-node file are byte-identical in
    construction. Returns the resolved path written.

    Mid-training safety, the reason this is a function both callers can
    use: trained_state_dict() only reads/detaches weights (no mutation),
    and optimizer backward hooks only fire during a backward() -- a
    save_every_n_steps save happens between optimizer steps, never
    mid-backward, so the hook race core.save.save_lora_checkpoint
    guards against (its temporary fused-hook removal) isn't reachable
    from this caller either.

    layout=None -> ProjectLayout.from_paths_module(); pass an explicit
    layout to redirect (tests pass a temp directory). Resolution goes
    through layout.resolve_safe_model_path(relative_path, "lora"): subfolders
    allowed, '..'/absolute rejected -- same sandbox as every other
    graph-reachable path."""
    from safetensors.torch import save_file

    state_dict = model.trained_state_dict()
    if not state_dict:
        raise ValueError("save_trained_weights: nothing to save -- "
                         "trained_state_dict() returned an empty dict.")
    # D3 health summary: per-group median ||dW||_F, printed and stored in
    # the file metadata so a weak/odd LoRA is visible at save time rather
    # than after a long render session. Never blocks the save -- a summary
    # that fails to compute must not lose the weights.
    from .lora_file import split_block_weights
    # The per-module block weights are lifted out of the tensors and into the
    # file's metadata (lora_file.py: a stray tensor key makes diffusers
    # refuse the whole file). The summary is computed on the full dict, which
    # still needs them.
    file_tensors, metadata = split_block_weights(state_dict)
    try:
        from .lora import lora_health_summary
        import json
        summary = lora_health_summary(state_dict)
        rows = " ".join(
            f"{g}={s['n']}x{ s['median_dW']:.4f}"
            for g, s in summary["groups"].items())
        print(f"  [LoRA health] {rows}")
        for w in summary["warnings"]:
            print(f"  [LoRA health] WARNING: {w}")
        metadata["lora_health"] = json.dumps(summary)
    except Exception as exc:  # noqa: BLE001 -- weights matter, the summary doesn't
        print(f"  [LoRA health] skipped ({type(exc).__name__}: {exc})")
    layout = layout or ProjectLayout.from_paths_module()
    resolved = layout.resolve_safe_model_path(relative_path, "lora")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    save_file(file_tensors, str(tmp), metadata=metadata or None)
    tmp.replace(resolved)
    return str(resolved)


def unfinished_path(relative_path: str, step: int | None) -> str:
    """`style/lora.safetensors` -> `style/lora_step5555.safetensors`.

    The step goes before the suffix and after the whole stem, so the file
    sorts next to the finished one and is obviously a snapshot of the same
    run. Only the final path component changes -- the directory part is
    left exactly as given, because it still has to pass
    resolve_safe_model_path's sandbox check unmodified. An unknown step
    (a producer that did not say) is named ``_unfinished`` rather than
    guessing a number."""
    from pathlib import PurePosixPath

    path = PurePosixPath(relative_path)
    tag = f"_step{step}" if step is not None else "_unfinished"
    return str(path.with_name(f"{path.stem}{tag}{path.suffix}"))


class LoRACheckpointSaverNode(CheckpointSaverNode):

    # After a user stop this is one of the nodes that still runs, to save
    # what training had reached (nodes/core.py Completion). It decides what
    # to do with it: see save_unfinished and build().
    ACCEPTS_UNFINISHED: ClassVar[bool] = True

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(
            name="model", type=TrainedWeightsExportable, required=True,
            doc="A trained model (e.g. SupervisedLoRATrainerNode's output) or a "
                "LoRAPhaseSplitNode's completed_generation snapshot -- both "
                "TrainedWeightsExportable, so either wires in here.",
        ),
        "relative_path": Port(
            name="relative_path", type=str, required=True, path_kind="lora_output",
            doc="Path relative to the configured LoRA directory, e.g. "
                "'style_v2/checkpoint_1000.safetensors'. Subfolders are created as "
                "needed. Absolute paths and '..' are rejected -- this field is reachable "
                "from the graph editor over the network, so it's sandboxed to the "
                "configured directory regardless of what's typed here.",
        ),
        "save_unfinished": Port(
            name="save_unfinished", type=bool, required=False, default=True,
            doc="When the run was stopped before it finished, still save what training "
                "had reached -- to the same path with `_step<N>` added to the file name "
                "(`lora.safetensors` -> `lora_step5555.safetensors`), so it never "
                "replaces or is mistaken for a finished LoRA. Off = a stop saves nothing "
                "here. Has no effect on a run that finished.",
        ),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, str]:
        self.validate_inputs(inputs)

        model = inputs["model"]
        if not isinstance(model, TrainedWeightsExportable):
            raise TypeError(
                f"LoRACheckpointSaverNode needs a TrainedWeightsExportable "
                f"(a trained model or a phase-split snapshot), got {type(model).__name__}."
            )
        relative_path = inputs["relative_path"]
        completion = self.upstream_completion("model")
        if completion.unfinished:
            save_unfinished = inputs.get(
                "save_unfinished", self.INPUTS["save_unfinished"].default)
            where = (f"step {completion.step}" if completion.step is not None
                     else "an unknown step")
            if completion.total is not None:
                where += f" of {completion.total}"
            if not save_unfinished:
                print(f"[LoRACheckpointSaverNode] the run stopped at {where}; "
                      f"save_unfinished is off, nothing saved")
                result = {"saved_path": ""}
                self.validate_outputs(result)
                return result
            if completion.step == 0:
                # Stopped before the first optimizer step: the adapter is
                # still its initial (zero-effect) self. A file of that is
                # not progress, and would only look like one.
                print("[LoRACheckpointSaverNode] the run stopped before its first "
                      "step; nothing trained, nothing saved")
                result = {"saved_path": ""}
                self.validate_outputs(result)
                return result
            relative_path = unfinished_path(relative_path, completion.step)
            print(f"[LoRACheckpointSaverNode] the run stopped at {where}; "
                  f"saving the unfinished LoRA as {relative_path}")
        saved_path = save_trained_weights(
            model, relative_path, inputs.get("project_layout"))

        result = {"saved_path": saved_path}
        self.validate_outputs(result)
        return result

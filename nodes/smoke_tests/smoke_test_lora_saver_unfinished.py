"""The LoRA saver and a stopped run: what file appears, and under what name.

Real safetensors I/O in a temporary LoRA directory, a tiny stand-in model
(only ``trained_state_dict`` matters to the saver). The completion state is
handed to the node the way the graph runtime hands it -- through a scoped
ExecutionContext -- so this exercises the node exactly as a run would, minus
the graph around it (that part is backend/tests/test_graph_runtime.py).

Cases: a finished run keeps the plain name; a stopped run saves
``<stem>_step<N><suffix>`` and leaves the plain name untouched;
``save_unfinished`` off saves nothing; a stop before the first step saves
nothing; the state is read from the wired ``model`` input only.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.components.layout import ProjectLayout
from nodes.core import COMPLETE, Completion, ExecutionContext
from nodes.model.handle import TrainedWeightsExportable
from nodes.model.lora_saver import LoRACheckpointSaverNode, unfinished_path


class _TinyModel(TrainedWeightsExportable):
    def trained_state_dict(self) -> dict:
        return {
            "lora_unet_root_proj.lora_down.weight": torch.ones(2, 4),
            "lora_unet_root_proj.lora_up.weight": torch.zeros(3, 2),
            "lora_unet_root_proj.alpha": torch.tensor(2.0),
        }


def _layout(root: Path) -> ProjectLayout:
    return ProjectLayout(
        comfy_dir=root, checkpoints_dir=root / "ckpt", loras_dir=root / "loras",
        datasets_dir=root / "ds", runs_dir=root / "runs")


def _run(layout, completion, relative_path="style/lora.safetensors", **extra):
    ctx = ExecutionContext().scoped({"model": completion})
    return LoRACheckpointSaverNode(ctx).build(
        model=_TinyModel(), relative_path=relative_path, project_layout=layout, **extra)


def _files(layout) -> list[str]:
    return sorted(str(p.relative_to(layout.loras_dir)) for p in layout.loras_dir.rglob("*")
                  if p.is_file())


def check_the_name_rule():
    print("[unfinished_path: step before the suffix, directory untouched]")
    assert unfinished_path("style/lora.safetensors", 5555) == "style/lora_step5555.safetensors"
    assert unfinished_path("lora.safetensors", 7) == "lora_step7.safetensors"
    assert unfinished_path("a/b/run.v2.safetensors", 12) == "a/b/run.v2_step12.safetensors"
    assert unfinished_path("style/lora.safetensors", None) == "style/lora_unfinished.safetensors"
    # Nothing in the name rule can walk out of the sandbox: only the last component changes.
    assert unfinished_path("x/lora.safetensors", 1).startswith("x/")
    print("    PASS")


def check_a_finished_run_keeps_its_plain_name():
    print("[complete input: the plain file name, as before]")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _layout(Path(tmp))
        out = _run(layout, COMPLETE)
        assert Path(out["saved_path"]).name == "lora.safetensors", out
        assert _files(layout) == ["style/lora.safetensors"], _files(layout)
        # An unfinished flag on a finished run has no effect either way.
        _run(layout, COMPLETE, save_unfinished=False)
        assert _files(layout) == ["style/lora.safetensors"]
    print("    PASS")


def check_a_stopped_run_saves_a_step_named_file():
    print("[unfinished input: <stem>_step<N><suffix>, the plain name not created]")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _layout(Path(tmp))
        out = _run(layout, Completion.unfinished_at(step=5555, total=8000))
        assert Path(out["saved_path"]).name == "lora_step5555.safetensors", out
        assert _files(layout) == ["style/lora_step5555.safetensors"], _files(layout)
        # A previously finished file at the plain name is never overwritten by a stop.
        plain = layout.loras_dir / "style" / "lora.safetensors"
        plain.write_bytes(b"finished earlier")
        _run(layout, Completion.unfinished_at(step=6000, total=8000))
        assert plain.read_bytes() == b"finished earlier"
        assert _files(layout) == ["style/lora.safetensors", "style/lora_step5555.safetensors",
                                  "style/lora_step6000.safetensors"], _files(layout)
        # The file is a real LoRA, readable back.
        from safetensors.torch import load_file
        tensors = load_file(str(layout.loras_dir / "style" / "lora_step5555.safetensors"))
        assert set(tensors) == set(_TinyModel().trained_state_dict()), set(tensors)
    print("    PASS")


def check_the_checkbox_off_saves_nothing():
    print("[save_unfinished off: no file, saved_path empty, still a valid result]")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _layout(Path(tmp))
        out = _run(layout, Completion.unfinished_at(step=5555), save_unfinished=False)
        assert out == {"saved_path": ""}, out
        assert _files(layout) == [], _files(layout)
    assert LoRACheckpointSaverNode.INPUTS["save_unfinished"].default is True
    print("    PASS")


def check_a_stop_before_the_first_step_saves_nothing():
    print("[step 0: the adapter is still its initial self, nothing is saved]")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _layout(Path(tmp))
        out = _run(layout, Completion.unfinished_at(step=0, total=100))
        assert out == {"saved_path": ""}, out
        assert _files(layout) == [], _files(layout)
    print("    PASS")


def check_an_unknown_step_is_named_not_guessed():
    print("[unfinished without a step: `_unfinished`, never an invented number]")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _layout(Path(tmp))
        out = _run(layout, Completion.unfinished_at())
        assert Path(out["saved_path"]).name == "lora_unfinished.safetensors", out
    print("    PASS")


def check_the_node_opts_in():
    print("[ACCEPTS_UNFINISHED: the saver is the node the runtime keeps running after a stop]")
    assert LoRACheckpointSaverNode.ACCEPTS_UNFINISHED is True
    print("    PASS")


def main():
    check_the_name_rule()
    check_a_finished_run_keeps_its_plain_name()
    check_a_stopped_run_saves_a_step_named_file()
    check_the_checkbox_off_saves_nothing()
    check_a_stop_before_the_first_step_saves_nothing()
    check_an_unknown_step_is_named_not_guessed()
    check_the_node_opts_in()
    print("\nSMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

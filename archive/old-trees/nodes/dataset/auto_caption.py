"""AutoCaptionNode: write captions into a managed dataset that lacks them.

Runs a caption model over the dataset's images and stores the result in the
dataset (manager/captioning.py has the mechanics and the backend registry).
Place it before the dataset source node and wire its ``dataset_root`` output to
that node's ``dataset_root`` input, so the source loads the captions this wrote.

The default backend is ``transformers`` -- a Hugging Face image-to-text model
(BLIP base unless ``model_id`` says otherwise) run on this machine. Other
backends are added with ``manager.captioning.register_captioner``; ``backend``
selects one by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..components.layout import ProjectLayout
from ..core import Node, Port


class AutoCaptionNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "dataset_root": Port(
            name="dataset_root", type=Path, required=True,
            doc="The same dataset name, passed through: wire it to the dataset "
                "source node so that node runs after the captions are written."),
        "summary": Port(
            name="summary", type=str, required=True,
            doc="One line: how many images were captioned, and why any were not."),
    }

    INPUTS: ClassVar[dict[str, Port]] = {
        "dataset_root": Port(
            name="dataset_root", type=Path, required=True, path_kind="dataset",
            doc="Dataset name from the library (sandboxed like the source node's)."),
        "backend": Port(
            name="backend", type=str, required=False, default="transformers",
            doc="Caption backend by name. 'transformers' = a local Hugging Face "
                "image-to-text model. More are added with "
                "manager.captioning.register_captioner."),
        "model_id": Port(
            name="model_id", type=str, required=False,
            default="Salesforce/blip-image-captioning-base",
            doc="Model for the backend (a Hugging Face id or a local folder). "
                "The default BLIP base writes short plain-English captions; "
                "downloads on first use, so the machine running this needs "
                "network access once."),
        "device": Port(
            name="device", type=str, required=False, default="auto",
            doc="'auto' = the XPU/CUDA device if there is one, else CPU; or a "
                "torch device string. Freed again when captioning ends."),
        "max_new_tokens": Port(
            name="max_new_tokens", type=int, required=False, default=40,
            doc="Longest caption the model may write. CLIP reads 77 tokens."),
        "overwrite": Port(
            name="overwrite", type=bool, required=False, default=False,
            doc="False = only images with an empty caption are captioned (so a "
                "re-run resumes and never replaces captions you wrote). True = "
                "replace every caption."),
        "allow_preview": Port(
            name="allow_preview", type=bool, required=False, default=True,
            doc="When an image's original file is gone, caption its 256px VAE "
                "preview from the dataset instead of skipping it."),
        "prefix": Port(
            name="prefix", type=str, required=False, default="",
            doc="Trigger words put before each caption, joined with ', '."),
        "suffix": Port(
            name="suffix", type=str, required=False, default="",
            doc="Words put after each caption, joined with ', '."),
        "lowercase": Port(
            name="lowercase", type=bool, required=False, default=False,
            doc="Lowercase the model's text (not the prefix/suffix)."),
        "limit": Port(
            name="limit", type=int, required=False, default=0,
            doc="Caption at most this many images this run; 0 = all. A cheap "
                "trial before the full pass."),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module()."),
    }

    def build(self, **inputs) -> dict[str, object]:
        self.validate_inputs(inputs)
        from manager.captioning import caption_dataset, make_captioner

        get = lambda name: inputs.get(name, self.INPUTS[name].default)  # noqa: E731
        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        root = layout.resolve_safe_dataset_path(str(inputs["dataset_root"]))
        captioner = make_captioner(
            get("backend"), model_id=get("model_id"), device=get("device"),
            max_new_tokens=get("max_new_tokens"))
        try:
            report = caption_dataset(
                root, captioner, overwrite=get("overwrite"),
                allow_preview=get("allow_preview"), prefix=get("prefix"),
                suffix=get("suffix"), lowercase=get("lowercase"),
                limit=get("limit") or None,
                should_cancel=self.context.should_cancel,
                on_progress=lambda done, total: print(
                    f"  [auto-caption] {done}/{total}", flush=True)
                if done % 25 == 0 else None)
        finally:
            captioner.close()
        for err in report.errors:
            print(f"  [auto-caption] failed: {err}")
        print(f"  [auto-caption] {report.summary()}")
        result = {"dataset_root": inputs["dataset_root"], "summary": report.summary()}
        self.validate_outputs(result)
        return result

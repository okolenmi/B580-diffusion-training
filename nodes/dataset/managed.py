"""ManagedDatasetSourceNode: wraps manager.loader.ManagedDatasetLoader.

Adapter only -- no dataset logic reimplemented here. ManagedDatasetLoader
already implements __iter__/__len__/invalidate_cache correctly (bucketing,
shard caching); this just makes it satisfy TrainingBatchSource.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..core import Port
from ..components.layout import ProjectLayout
from .handle import TrainingBatchSource
from .node import DataSourceNode
from .timestep_modes import T_MODES


class ManagedDatasetBatchSource(TrainingBatchSource):

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self) -> Iterator[dict]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def invalidate(self) -> None:
        self._loader.invalidate_cache()


class ManagedDatasetSourceNode(DataSourceNode):
    """Streams batches from a manager-managed dataset (safetensors shards + sqlite index)."""

    INPUTS: ClassVar[dict[str, Port]] = {
        "dataset_root": Port(name="dataset_root", type=Path, required=True, path_kind="dataset",
                              doc="Dataset name from the library. Absolute paths and '..' are rejected -- "
                                  "this field is reachable from the graph editor over the network, so "
                                  "it's sandboxed to the configured datasets directory."),
        "set_identifier": Port(name="set_identifier", type=Any, required=False, default=None,
                                doc="Training-set name or ID; None = every trajectory in the dataset."),
        "shuffle": Port(name="shuffle", type=bool, required=False, default=True),
        "batch_size": Port(name="batch_size", type=int, required=False, default=1),
        "use_dataset_cfg": Port(name="use_dataset_cfg", type=bool, required=False, default=True),
        "t_low": Port(name="t_low", type=int, required=False, default=1,
                      doc="Only affects 'lora_raw'-format trajectories (see manager/builder.py's "
                          "run_lora_ingestion_task) -- every other format has its own t baked in."),
        "t_high": Port(name="t_high", type=int, required=False, default=999),
        "t_mode": Port(name="t_mode", type=str, required=False, default="uniform",
                       choices=T_MODES,
                       doc="Same distributions core.noise_schedule.sample_timestep implements."),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainingBatchSource]:
        self.validate_inputs(inputs)
        from manager.loader import ManagedDatasetLoader

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        loader = ManagedDatasetLoader(
            dataset_root=layout.resolve_safe_dataset_path(str(inputs["dataset_root"])),
            set_identifier=inputs.get("set_identifier", self.INPUTS["set_identifier"].default),
            shuffle=inputs.get("shuffle", self.INPUTS["shuffle"].default),
            batch_size=inputs.get("batch_size", self.INPUTS["batch_size"].default),
            use_dataset_cfg=inputs.get("use_dataset_cfg", self.INPUTS["use_dataset_cfg"].default),
            t_low=inputs.get("t_low", self.INPUTS["t_low"].default),
            t_high=inputs.get("t_high", self.INPUTS["t_high"].default),
            t_mode=inputs.get("t_mode", self.INPUTS["t_mode"].default),
        )
        result = {"batches": ManagedDatasetBatchSource(loader)}
        self.validate_outputs(result)
        return result

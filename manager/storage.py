"""Tensor sharding logic — packs and unpacks training data."""

from pathlib import Path
import torch
import json
from safetensors.torch import save_file
from safetensors import safe_open


class ShardWriter:
    """Handles packing of trajectories into physical shards."""
    
    @staticmethod
    def pack_shard(file_path: Path, samples: list[dict]):
        """Helper to quickly pack a list of samples into a shard file."""
        writer = ShardWriter(file_path)
        writer.add_trajectory(samples)
        return writer.write()

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.samples_buffer = {}
        self.metadata = {}
        self.current_index = 0

    def add_trajectory(self, trajectory_samples: list[dict]):
        """
        Add a full trajectory of samples to the buffer.
        Returns the starting index in the shard for this trajectory.
        """
        start_idx = self.current_index
        for i, s in enumerate(trajectory_samples):
            idx = self.current_index + i
            xt = s["x_t"].contiguous()
            self.samples_buffer[f"x_t_{idx}"] = xt
            # Clone to break shared memory — safetensors rejects aliased tensors
            self.samples_buffer[f"target_{idx}"] = s["target"].contiguous().clone()
            self.samples_buffer[f"t_{idx}"] = torch.tensor([s["t"]], dtype=torch.int32)
            
            if s.get("target_p") is not None:
                self.samples_buffer[f"target_p_{idx}"] = s["target_p"].contiguous()
            if s.get("target_n") is not None:
                self.samples_buffer[f"target_n_{idx}"] = s["target_n"].contiguous()

            self.metadata[f"meta_{idx}"] = {
                "at": s.get("at", 1.0), "st": s.get("st", 0.0)
            }
        
        self.current_index += len(trajectory_samples)
        return start_idx

    def add_compressed_trajectory(self, xt_seq: torch.Tensor, p_seq: torch.Tensor, n_seq: torch.Tensor, 
                                  t_grid: list[int], meta_list: list[dict]):
        """
        Store entire trajectory as stacked tensors to minimize key overhead.
        xt_seq: (Steps, 4, H, W)
        p_seq, n_seq: (Steps, 4, H, W)
        """
        traj_id = self.current_index
        self.samples_buffer[f"traj_{traj_id}_xt"] = xt_seq.contiguous()
        self.samples_buffer[f"traj_{traj_id}_p"] = p_seq.contiguous()
        self.samples_buffer[f"traj_{traj_id}_n"] = n_seq.contiguous()
        self.samples_buffer[f"traj_{traj_id}_t"] = torch.tensor(t_grid, dtype=torch.int32)
        
        self.metadata[f"traj_meta_{traj_id}"] = meta_list
        self.current_index += 1
        return traj_id

    def write(self):
        """Finalize and write to disk."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        str_metadata = {k: json.dumps(v) for k, v in self.metadata.items()}
        save_file(self.samples_buffer, str(self.file_path), metadata=str_metadata)
        return self.current_index, self.file_path.stat().st_size


class ShardLoader:
    """Lazily loads samples from a shard — reads tensors on demand via safe_open."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._file = None
        self._meta = None
        self._keys: set[str] | None = None

    def load(self):
        """Open the shard file. Tensors are NOT loaded into RAM — read on demand."""
        self._file = safe_open(str(self.file_path), framework="pt", device="cpu")
        raw_meta = self._file.metadata()
        self._meta = {k: json.loads(v) for k, v in raw_meta.items()} if raw_meta else {}
        self._keys = set(self._file.keys())
        return self

    def _ensure_open(self):
        if self._file is None:
            self.load()

    def _has_key(self, key: str) -> bool:
        self._ensure_open()
        return key in self._keys

    def get_tensor(self, key: str):
        self._ensure_open()
        return self._file.get_tensor(key)

    def get_sample(self, index: int):
        self._ensure_open()
        m = self._meta.get(f"meta_{index}", {})
        out = {
            "x_t": self.get_tensor(f"x_t_{index}"),
            "target": self.get_tensor(f"target_{index}"),
            "t": self.get_tensor(f"t_{index}").item(),
            "at": m.get("at", 1.0),
            "st": m.get("st", 0.0)
        }
        if self._has_key(f"target_p_{index}"):
            out["target_p"] = self.get_tensor(f"target_p_{index}")
        if self._has_key(f"target_n_{index}"):
            out["target_n"] = self.get_tensor(f"target_n_{index}")
        return out

    def get_trajectory_samples(self, start_index: int, count: int) -> list[dict]:
        self._ensure_open()
        return [self.get_sample(i) for i in range(start_index, start_index + count)]

    def get_compressed_trajectory(self, traj_id: int) -> list[dict]:
        """Load a compressed trajectory and expand it into a list of samples."""
        self._ensure_open()
        xt_key = f"traj_{traj_id}_xt"
        if not self._has_key(xt_key):
            raise KeyError(f"Compressed trajectory {traj_id} not found in {self.file_path}")

        xts = self.get_tensor(xt_key)
        ps = self.get_tensor(f"traj_{traj_id}_p")
        ns = self.get_tensor(f"traj_{traj_id}_n")
        ts = self.get_tensor(f"traj_{traj_id}_t")
        meta = self._meta.get(f"traj_meta_{traj_id}", [])

        samples = []
        for i in range(len(ts)):
            m = meta[i] if i < len(meta) else {}
            samples.append({
                "x_t": xts[i:i+1],
                "target_p": ps[i:i+1],
                "target_n": ns[i:i+1],
                "t": ts[i].item(),
                "at": m.get("at", 1.0),
                "st": m.get("st", 0.0)
            })
        return samples

    def close(self):
        self._file = None
        self._meta = None
        self._keys = None

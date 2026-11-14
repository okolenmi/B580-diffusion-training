"""Caption dropout: the knob that stops cross-attention going unconstrained.

The bug it exists for, measured on real SDXL weights: a dataset whose
captions are all one value (`test2`: 204 images, 204 empty captions)
presents a CONSTANT (ctx, y) every step. A cross-attention projection
`W_k . c` whose input never changes gets zero gradient from any
loss-dependent path, so its key/query/value projections are never shaped
to interpret a context -- unconstrained, not underfit. The resulting LoRA
falls apart when conditioning changes at render, and the *least*
disturbing change was the worst one (CLIP-L-only moved the context 0.081
from the training value, CLIP-G-only 0.307, full 0.318).

These checks are logic-level and tiny-CPU: they pin that dropout actually
varies the captions a batch is conditioned on, that it draws only from the
dataset's own captions, and that the default is off so no existing run
changes behaviour. The image-level benefit needs a real render, which is
the maintainer's call.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from manager.loader import ManagedDatasetLoader

REPO = Path(__file__).resolve().parents[2]


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _loader(caption_dropout: float, dataset: str = "test2") -> ManagedDatasetLoader:
    return ManagedDatasetLoader(dataset_root=REPO / "datasets" / dataset,
                                batch_size=2, caption_dropout=caption_dropout)


def _stub_samples(prompts):
    """Materialized-batch-shaped samples: what _merge_samples expects."""
    return [{"prompt": p, "neg_prompt": "", "x_t": torch.zeros(1, 4, 8, 8),
             "target": torch.zeros(1, 4, 8, 8), "t": 1, "seed": 0,
             "metadata": {}, "traj_type": "eps"} for p in prompts]


def check_default_is_off():
    print("[default: caption_dropout=0.0 leaves every caption untouched]")
    loader = _loader(0.0)
    check(loader.caption_dropout == 0.0, loader.caption_dropout)
    prompts = ["alpha", "beta", "gamma"]
    out = loader._merge_samples(_stub_samples(prompts))
    check(out["prompts"] == prompts, out["prompts"])
    check(out["prompt"] == prompts[0], out["prompt"])
    print("    PASS")


def check_rejects_an_out_of_range_value():
    print("[caption_dropout outside 0..1 is refused at construction]")
    for bad in (-0.1, 1.5):
        try:
            _loader(bad)
        except ValueError:
            pass
        else:
            check(False, f"caption_dropout={bad} was accepted")
    print("    PASS")


def check_dropout_actually_varies_the_captions():
    print("[dropout > 0 changes the captions a batch is conditioned on]")
    loader = _loader(1.0, dataset="multi-caption")
    samples = loader._load_all_samples()
    loader._samples = samples
    loader._caption_pool = sorted({s["prompt"] for s in samples})
    check(len(loader._caption_pool) >= 2,
          f"multi-caption should hold several captions, got "
          f"{len(loader._caption_pool)}")
    prompts = [samples[0]["prompt"]] * 4      # one caption, four samples
    random.seed(0)
    dropped = loader._merge_samples(_stub_samples(prompts))["prompts"]
    check(len(set(dropped)) > 1,
          f"caption_dropout=1.0 left a uniform batch uniform: {dropped}")
    check(set(dropped) <= set(loader._caption_pool),
          f"dropout invented a caption: {set(dropped) - set(loader._caption_pool)}")
    print(f"    PASS ({len(loader._caption_pool)} captions in the dataset; a "
          f"uniform batch became {len(set(dropped))} distinct)")


def check_prompt_and_prompts_stay_consistent():
    print("[after dropout, `prompt` and `prompts` cannot disagree]")
    loader = _loader(1.0, dataset="multi-caption")
    samples = loader._load_all_samples()
    loader._samples = samples
    loader._caption_pool = sorted({s["prompt"] for s in samples})
    random.seed(1)
    out = loader._merge_samples(_stub_samples([samples[0]["prompt"]] * 4))
    check(out["prompt"] == out["prompts"][0],
          f"batch-wide prompt {out['prompt']!r} != first per-sample "
          f"{out['prompts'][0]!r} -- the trainers route on whether `prompts` "
          f"is mixed, so a stale `prompt` makes the two paths disagree")
    print("    PASS")


def check_a_single_caption_dataset_is_a_documented_no_op():
    print("[one distinct caption: nothing to substitute, so it returns unchanged]")
    loader = _loader(1.0)                       # test2: one caption (empty)
    samples = loader._load_all_samples()
    loader._samples = samples
    loader._caption_pool = sorted({s["prompt"] for s in samples})
    check(len(loader._caption_pool) == 1, loader._caption_pool)
    before = ["alpha", "beta"]
    after = loader._merge_samples(_stub_samples(before))["prompts"]
    check(after == before, (before, after))
    print("    PASS (no-op rather than pretending to work)")


def check_node_exposes_it_defaulting_off():
    print("[the dataset node exposes caption_dropout, default 0.0]")
    from nodes.dataset.managed import ManagedDatasetSourceNode
    port = ManagedDatasetSourceNode.INPUTS["caption_dropout"]
    check(port.default == 0.0, port.default)
    print("    PASS")


def main():
    check_default_is_off()
    check_rejects_an_out_of_range_value()
    check_dropout_actually_varies_the_captions()
    check_prompt_and_prompts_stay_consistent()
    check_a_single_caption_dataset_is_a_documented_no_op()
    check_node_exposes_it_defaulting_off()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
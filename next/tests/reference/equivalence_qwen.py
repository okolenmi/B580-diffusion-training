"""Our Qwen3 encoder against Hugging Face's ``Qwen3Model`` with the same random weights (needs ``transformers``).

    python -m next.tests.reference.equivalence_qwen
"""
import torch
from transformers import Qwen3Config, Qwen3Model

from ...families.anima.text import QwenConfig, QwenEncoder

torch.manual_seed(0)
cfg = QwenConfig(vocab=500, width=64, layers=3, heads=4, kv_heads=2, head_dim=32, mlp_width=96)
ref = Qwen3Model(Qwen3Config(vocab_size=500, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=128,
                             attn_implementation="eager")).eval()
for p in ref.parameters():
    p.data.normal_(0, 0.1)
mine = QwenEncoder(cfg).eval()
print(mine.load_state_dict({f"model.{k}": v for k, v in ref.state_dict().items()}))
ids = torch.randint(0, 500, (2, 11))
with torch.no_grad():
    a = ref(input_ids=ids).last_hidden_state
    b = mine(ids)
print("max abs diff", (a - b).abs().max().item(), "scale", a.abs().mean().item())
mask = torch.ones(2, 11, dtype=torch.long); mask[1, 7:] = 0
with torch.no_grad():
    a = ref(input_ids=ids, attention_mask=mask).last_hidden_state
    b = mine(ids, mask)
print("padded max abs diff (real rows)", (a[1, :7] - b[1, :7]).abs().max().item())

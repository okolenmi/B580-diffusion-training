# LoRA training guide (SDXL, B580)

Short and honest about evidence. Every recommendation below is labelled:
*proven on the 1-image A1 case*, *pending the 5-arm B study*, or
*trainer default (unchanged)*. Do not generalize the A1 rows beyond what
they are until B reports.

## Starting settings

| setting | value | evidence |
|---|---|---|
| LR schedule | **constant** 1e-4 | A1: cosine-to-5% applies ~0.52x the nominal LR and undertrains 1.5-1.8x (forensics match the renders). Pending B factorial |
| targets | **kohya_plus** (726) | A1 renders best on captionless data. Pending B + cond-path check |
| rank / alpha | 16 / 16 for tests; 64 / 32–48 for real runs | rank gates OOM on 12 GB (kohya rank 32 dies ~40 steps at 512 px); ours unspecified -- start 64/32 |
| batch | 4 (2-4 free; 8 GPU-bound) | measured L4 |
| bucketing | off unless multi-resolution | x32 free on the measured set, default stays off (L2) |
| graph capture | off unless you measured room | +37% for +1.5 GB, managed route (L5.3) |

Trainer defaults are unchanged (cosine, attention+cond 564, no capture):
existing graphs/configs behave exactly as before. The C3 decision note
(`docs/decisions/`, when written) proposes defaults for *new* runs.

## Effective LR

Cosine and poly are not "the LR you typed": the run applies the schedule
mean (cosine-to-5% ≈ 0.52x). Every trainer build prints
`LR: <kind> peak … mean … (…% of peak)` and warns below 70%. If you see
the warning on cosine, it is expected; anywhere else, fix the schedule,
not the LR.

## Reading the health summary

Every save prints per-group median `||dW||` (`attn_input/middle/output`,
`ff/proj`, cond path) and stores it in the file metadata (`lora_health`).
Two warnings matter:

- `attn_output > 1.8x attn_input`: the output projections carry the
  adaptation (the attention-only signature was 2.0x).
- any group `> 5x` the attention reference: something pooled on the
  global conditioning path -- check what and why.

Same table offline: `archive/bad-LoRA-investigation/lora_forensics.py`.

## Compatibility of cond-path modules

`kohya_plus` adds `time_embed.*` / `label_emb.*` keys. ComfyUI applies
them (that is where the renders come from). diffusers ≥0.41 + peft loads
the file including those keys, mapped to `time_embedding` and
`add_embedding` linears -- verified, nothing ignored, no error. Whether
the cond modules bake in dataset constants (caption/size) is the open
B3 check: compare arm 4 vs arm 5 renders before making plus the default.

## Records (not pasted here)

- `archive/bad-LoRA-investigation/MEASURED-targets-and-schedule.md` (A1)
- `archive/bad-LoRA-investigation/MEASURED-original-symptom.md` (B, when done)
- `archive/bad-LoRA-investigation/MEASURED-kohya-on-xpu.md` (kohya on B580)
- `shapes diversity problem 2/STATUS.md` + `MEASURED-launch-bound.md` (speed)

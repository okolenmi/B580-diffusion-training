# STATUS: launch-bound task (L1–L5) — per-item report

Task: `TASK-launch-bound.md` in this folder. Hardware throughout: Intel
Arc B580, 12,216 MB, torch 2.12.1+xpu, SDXL LoRA, managed route,
dataset `non-square`, unless noted. One run at a time.

| Item | Status | Commit | Measured before → after |
|---|---|---|---|
| L1 true-size conditioning for bucketed samples | **done** | `5452156` | bucketed samples conditioned by mask-derived true size, not the bucket |
| L2a pad-fraction report at build | **done** | `ad22f94` | per-sample pad fraction + distribution logged once at build |
| L2b,c bucketing quality + multiple | **done** | `2205d15`, voided by `97c819b`, re-run `e98cde7` | x16/x32 indistinguishable from off on paired 300-step fit metric (Δ within ±0.00026 vs +0.000443 seed control); only x48 worse (+0.002302, 5.2x control); mechanism = pad fraction, threshold between 25% and 51%; x32 1.50x steps/s, peak unchanged; **default stays off** |
| L2 x64 separator arm | **done** | `a72b132` | x64 fine — kills the SDXL-bucket-affinity hypothesis |
| LossPhase bucketing-mask fix (found by L2) | **done** | `0a7f6a2` | managed route trained on padding at valid-fraction rate before; masked after |
| L3 primitive-cache-thrash detector | **done** | `f4faf4b` | revisit-vs-repeat warning in `MonitoringPhase`, synthetic-step-time test |
| L4 premise (batch scaling) | **measured, code not built** | `64186e3` | step time flat batch 1→4 (0.988→0.996 steps/s, 4.03x images/s); batch 8 GPU-bound (1.55 s/step, +960 MB, 640 MB drift); batch 4 is the sweet spot. Heterogeneous-caption cost unmeasured (one-caption dataset) |
| L5.1 checkpointing fraction | **done 2026-10-09** | (this commit) | non-monotone: 0.5 → +17–24% steps/s for one-time +2.1 GB (flat over 300 steps); 0.25 → 1.8x slower steps (allocator-pressure regime); 0.0 OOMs; loss identical all arms; 1.0 default stands |
| L5.2 LoRA-branch launches | **measured, not recommended, not built** | — | best shippable variant −0.5%; −7.2% costs up to 2.27% adapter-delta error; −22.8% needs an invalidation design that does not exist (recorded in `docs/known-issues/deferred.md`) |
| L5.3 XPU graph capture | **measured, not integrated** | — | 2.04–2.08x replay vs production eager (MATH/EFFICIENT SDPA; FLASH cannot capture — SYCL limit); per-shape pools ~460–520 MB; optimizer step, multi-shape, dropout RNG open |
| Per-axis pad report (`PENDING-per-axis-pad-report.md`) | **done 2026-10-09** | (this commit) | modal sides + divisibility + untouched/both-axes counts in the build report; x32 keeps 204/204 modal widths, x24 keeps 0, pads both axes on 254/273 |

Open, in cheapest-first order: per-axis pad report (one pass over
trajectories, spec complete) → `--attn-ckpt-fraction` sweep → L4
per-sample captions (needs a multi-caption dataset to evaluate) → L5.3
integration (one graph per bucket shape, optimizer step, static
conditioning). L5.2 stays unbuilt by measurement, not by neglect.

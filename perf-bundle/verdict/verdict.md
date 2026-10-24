# Acceptance verdict

24 passed, 7 failed, 1 for your eyes, 0 skipped.

| suite | TESTING | result |
|---|---|---|
| Speed of native, int8, graph and attention modes | 23, 25, 26, 28 | FAIL |
| Allocator A/B | 24 | LOOK |

## Speed of native, int8, graph and attention modes (TESTING 23, 25, 26, 28): FAIL

22 passed, 7 failed, 0 to look at, 0 skipped.

| check | result | detail |
|---|---|---|
| native graph finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  / step 25/28  loss 0.1117  5839 ms  epoch 1  [2x64x128 new]  graphs 2 (18 replays)  device 9046 MB (live 5549, step peak 6305, cached 7542) / step 26/28  loss 0.0226  5931 ms  epoch 1  [2x64x128]  graphs 2 (18 replays)  device 9054 MB (live 5549, step peak 6305, cached 7542) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:22:00 7363:7363 ActivityProfilerController.cpp:455] profiler_stop / [  73.7s] DiffusionTrainerNode: 70.1s  allocated 5549, reserved 7542, device 9049 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| int8 graph finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  /        673  other programs / step 26/28  loss 0.0226  6918 ms  epoch 1  [2x64x128]  graphs 2 (18 replays)  device 7506 MB (live 3489, step peak 4246, cached 5708) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:25:22 7512:7512 ActivityProfilerController.cpp:455] profiler_stop / [  96.0s] DiffusionTrainerNode: 80.9s  allocated 3489, reserved 5708, device 7506 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| int8 dx-int8 graph finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  /        673  other programs / step 26/28  loss 0.0226  7463 ms  epoch 1  [2x64x128]  graphs 2 (18 replays)  device 7519 MB (live 3489, step peak 4253, cached 5710) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:29:03 7724:7724 ActivityProfilerController.cpp:455] profiler_stop / [ 103.5s] DiffusionTrainerNode: 88.4s  allocated 3489, reserved 5710, device 7519 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| native graph attention-math_bf16 finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  / step 25/28  loss 0.1117  6272 ms  epoch 1  [2x64x128 new]  graphs 2 (18 replays)  device 8749 MB (live 5549, step peak 6305, cached 7142) / step 26/28  loss 0.0226  6385 ms  epoch 1  [2x64x128]  graphs 2 (18 replays)  device 8717 MB (live 5549, step peak 6305, cached 7142) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:34:50 7967:7967 ActivityProfilerController.cpp:455] profiler_stop / [  73.6s] DiffusionTrainerNode: 69.9s  allocated 5549, reserved 7142, device 8701 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| native graph attention-chunked_bf16 finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  / step 25/28  loss 0.1118  6369 ms  epoch 1  [2x64x128 new]  graphs 2 (4 replays)  device 8928 MB (live 5548, step peak 6304, cached 7362) / step 26/28  loss 0.0226  6421 ms  epoch 1  [2x64x128]  graphs 2 (4 replays)  device 8948 MB (live 5548, step peak 6304, cached 7362) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:36:14 8171:8171 ActivityProfilerController.cpp:455] profiler_stop / [  81.0s] DiffusionTrainerNode: 77.1s  allocated 5548, reserved 7362, device 8919 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| int8 graph attention-chunked_bf16 finished | FAIL | the run did not finish: run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  /        574  other programs / step 26/28  loss 0.0226  7486 ms  epoch 1  [2x64x128]  graphs 2 (18 replays)  device 7253 MB (live 3489, step peak 4246, cached 5548) / perf: window done, processing the profile (single-threaded, this takes a while on a real model) ... / USDT:2026-10-24 20:37:58 8252:8252 ActivityProfilerController.cpp:455] profiler_stop / [ 103.3s] DiffusionTrainerNode: 87.0s  allocated 3489, reserved 5548, device 7253 MB / run: failed: RuntimeError: Profiling information is unavailable for events returned from a submission to a queue in the recording state.  |
| chunked_bf16 attention beats math in a graph | FAIL | median step 1036 ms against 849 ms: x1.22, over x1.00 (from: TESTING 28) |
| native eager finished | PASS | the run finished |
| int8 eager finished | PASS | the run finished |
| int8 dx-int8 eager finished | PASS | the run finished |
| native eager loss finite | PASS | 28 losses, all finite |
| native graph loss finite | PASS | 26 losses, all finite |
| int8 eager loss finite | PASS | 28 losses, all finite |
| int8 graph loss finite | PASS | 26 losses, all finite |
| int8 dx-int8 eager loss finite | PASS | 28 losses, all finite |
| int8 dx-int8 graph loss finite | PASS | 26 losses, all finite |
| native graph attention-math_bf16 loss finite | PASS | 26 losses, all finite |
| native graph attention-chunked_bf16 loss finite | PASS | 26 losses, all finite |
| int8 graph attention-chunked_bf16 loss finite | PASS | 26 losses, all finite |
| int8 holds less memory than native | PASS | peak_mb 4244 against 6303 MB: 2059 MB lower (wanted 1500) |
| int8 graph within 15% of native graph | PASS | median step 952 ms against 849 ms: x1.12 (limit x1.15) |
| native graph no slower than eager | PASS | median step 849 ms against 949 ms: x0.89 (limit x1.03) |
| int8 graph no slower than eager | PASS | median step 952 ms against 1330 ms: x0.72 (limit x1.03) |
| int8 graph decision | PASS | (2, 4, 100, 64): replay 914 ms against 1267 ms eager kept |
| native graph decision | PASS | (2, 4, 100, 64): replay 810 ms against 918 ms eager kept |
| int8 gradient path int8 no slower than dense | PASS | median step 929 ms against 952 ms: x0.98 (limit x1.00) |
| math_bf16 attention beats math in a graph | PASS | median step 691 ms against 849 ms: x0.81 (limit x1.00) |
| int8 loss close to native | PASS | mean loss over the last 10 steps 0.1050 against 0.1049 (0.1%, limit 15%) |
| chunked_bf16 loss close to math | PASS | mean loss over the last 10 steps 0.1268 against 0.1268 (0.0%, limit 10%) |

## Allocator A/B (TESTING 24): LOOK

2 passed, 0 failed, 1 to look at, 0 skipped.

| check | result | detail |
|---|---|---|
| allocator setting changes the peak? | LOOK | device_mb peaks native expandable-segments 7800 MB, native no-expandable-segments 8452 MB. Equal peaks mean the setting has no effect on this workload (TESTING 24). |
| native expandable-segments finished | PASS | the run finished |
| native no-expandable-segments finished | PASS | the run finished |

## Needs your eyes

- [ ] Allocator A/B: allocator setting changes the peak?. device_mb peaks native expandable-segments 7800 MB, native no-expandable-segments 8452 MB. Equal peaks mean the setting has no effect on this workload (TESTING 24).

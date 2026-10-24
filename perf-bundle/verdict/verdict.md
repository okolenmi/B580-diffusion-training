# Acceptance verdict

30 passed, 1 failed, 1 for your eyes, 0 skipped.

| suite | TESTING | result |
|---|---|---|
| Speed of native, int8, graph and attention modes | 23, 25, 26, 28 | FAIL |
| Allocator A/B | 24 | LOOK |

## Speed of native, int8, graph and attention modes (TESTING 23, 25, 26, 28): FAIL

28 passed, 1 failed, 0 to look at, 0 skipped.

| check | result | detail |
|---|---|---|
| chunked_bf16 attention beats math in a graph | FAIL | median step 968 ms against 916 ms: x1.06, over x1.00 (from: TESTING 28) |
| native eager finished | PASS | the run finished |
| native graph finished | PASS | the run finished |
| int8 eager finished | PASS | the run finished |
| int8 graph finished | PASS | the run finished |
| int8 dx-int8 eager finished | PASS | the run finished |
| int8 dx-int8 graph finished | PASS | the run finished |
| native graph attention-math_bf16 finished | PASS | the run finished |
| native graph attention-chunked_bf16 finished | PASS | the run finished |
| int8 graph attention-chunked_bf16 finished | PASS | the run finished |
| native eager loss finite | PASS | 28 losses, all finite |
| native graph loss finite | PASS | 28 losses, all finite |
| int8 eager loss finite | PASS | 28 losses, all finite |
| int8 graph loss finite | PASS | 28 losses, all finite |
| int8 dx-int8 eager loss finite | PASS | 28 losses, all finite |
| int8 dx-int8 graph loss finite | PASS | 28 losses, all finite |
| native graph attention-math_bf16 loss finite | PASS | 28 losses, all finite |
| native graph attention-chunked_bf16 loss finite | PASS | 28 losses, all finite |
| int8 graph attention-chunked_bf16 loss finite | PASS | 28 losses, all finite |
| int8 holds less memory than native | PASS | peak_mb 4244 against 6303 MB: 2059 MB lower (wanted 1500) |
| int8 graph within 15% of native graph | PASS | median step 1042 ms against 916 ms: x1.14 (limit x1.15) |
| native graph no slower than eager | PASS | median step 916 ms against 1076 ms: x0.85 (limit x1.03) |
| int8 graph no slower than eager | PASS | median step 1042 ms against 1673 ms: x0.62 (limit x1.03) |
| int8 graph decision | PASS | (2, 4, 100, 64): replay 992 ms against 1492 ms eager kept |
| native graph decision | PASS | (2, 4, 100, 64): replay 869 ms against 1076 ms eager kept |
| int8 gradient path int8 no slower than dense | PASS | median step 956 ms against 1042 ms: x0.92 (limit x1.00) |
| math_bf16 attention beats math in a graph | PASS | median step 713 ms against 916 ms: x0.78 (limit x1.00) |
| int8 loss close to native | PASS | mean loss over the last 10 steps 0.1050 against 0.1050 (0.0%, limit 15%) |
| chunked_bf16 loss close to math | PASS | mean loss over the last 10 steps 0.1050 against 0.1050 (0.0%, limit 10%) |

## Allocator A/B (TESTING 24): LOOK

2 passed, 0 failed, 1 to look at, 0 skipped.

| check | result | detail |
|---|---|---|
| allocator setting changes the peak? | LOOK | device_mb peaks native expandable-segments 7922 MB, native no-expandable-segments 8474 MB. Equal peaks mean the setting has no effect on this workload (TESTING 24). |
| native expandable-segments finished | PASS | the run finished |
| native no-expandable-segments finished | PASS | the run finished |

## Needs your eyes

- [ ] Allocator A/B: allocator setting changes the peak?. device_mb peaks native expandable-segments 7922 MB, native no-expandable-segments 8474 MB. Equal peaks mean the setting has no effect on this workload (TESTING 24).

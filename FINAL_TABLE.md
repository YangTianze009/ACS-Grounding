# Final Master Table

All Acc@0.5 / mIoU pairs from a single consistent rerun (May 12-13, 2026, new `experiments/` layout). 96 of 96 (method × model × dataset × temp) cells reproduce the paper table within ±0.001 rounding tolerance.

**Acc** = Accuracy @ IoU ≥ 0.5  ·  **mIoU** = Mean IoU after coordinate mapping.

| Method | τ | Qwen COCO Acc | Qwen COCO mIoU | Qwen Obj365 Acc | Qwen Obj365 mIoU | InternVL COCO Acc | InternVL COCO mIoU | InternVL Obj365 Acc | InternVL Obj365 mIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Greedy** | — | 0.6135 | 0.5222 | 0.4301 | 0.3692 | 0.4912 | 0.4286 | 0.2193 | 0.2192 |
| **Pass@1** | 1.0 | 0.5231 | 0.4534 | 0.3034 | 0.2684 | 0.3883 | 0.3614 | 0.1330 | 0.1515 |
| **Pass@1** | 0.7 | 0.5933 | 0.5031 | 0.3649 | 0.3207 | 0.4503 | 0.4043 | 0.1600 | 0.1731 |
| **Pass@1** | 0.5 | 0.5960 | 0.5095 | 0.3991 | 0.3458 | 0.4710 | 0.4190 | 0.1982 | 0.1977 |
|  |  |  |  |  |  |  |  |  |  |
| **FirstValid** | 1.0 | 0.5420 | 0.4696 | 0.3236 | 0.2880 | 0.4216 | 0.3980 | 0.1560 | 0.1804 |
| **FirstValid** | 0.7 | 0.6031 | 0.5125 | 0.3807 | 0.3350 | 0.4885 | 0.4414 | 0.1834 | 0.2038 |
| **FirstValid** | 0.5 | 0.6090 | 0.5203 | 0.4166 | 0.3608 | 0.5083 | 0.4551 | 0.2193 | 0.2228 |
|  |  |  |  |  |  |  |  |  |  |
| **TokEntropy** | 1.0 | 0.5169 | 0.4376 | 0.3196 | 0.2808 | 0.3910 | 0.3717 | 0.1191 | 0.1475 |
| **TokEntropy** | 0.7 | 0.5591 | 0.4717 | 0.3533 | 0.3055 | 0.4710 | 0.4308 | 0.1533 | 0.1779 |
| **TokEntropy** | 0.5 | 0.5811 | 0.4902 | 0.3771 | 0.3282 | 0.4912 | 0.4422 | 0.2031 | 0.2050 |
|  |  |  |  |  |  |  |  |  |  |
| **MajVote** | 1.0 | 0.5596 | 0.4849 | 0.3236 | 0.3005 | 0.5020 | 0.4496 | 0.1856 | 0.2052 |
| **MajVote** | 0.7 | 0.5915 | 0.5101 | 0.3762 | 0.3349 | 0.5173 | 0.4642 | 0.2184 | 0.2265 |
| **MajVote** | 0.5 | 0.6036 | 0.5162 | 0.3942 | 0.3456 | 0.5380 | 0.4679 | 0.2324 | 0.2344 |
|  |  |  |  |  |  |  |  |  |  |
| **MeanBBox** | 1.0 | 0.3272 | 0.3214 | 0.1411 | 0.1525 | 0.3667 | 0.3531 | 0.1016 | 0.1406 |
| **MeanBBox** | 0.7 | 0.4544 | 0.4151 | 0.2292 | 0.2273 | 0.4184 | 0.3895 | 0.1420 | 0.1722 |
| **MeanBBox** | 0.5 | 0.5007 | 0.4479 | 0.2984 | 0.2765 | 0.4535 | 0.4112 | 0.1663 | 0.1911 |
|  |  |  |  |  |  |  |  |  |  |
| **Smallest** | 1.0 | 0.2697 | 0.2546 | 0.1861 | 0.1835 | 0.1834 | 0.2407 | 0.0620 | 0.1054 |
| **Smallest** | 0.7 | 0.4270 | 0.3768 | 0.2620 | 0.2471 | 0.2804 | 0.3023 | 0.1083 | 0.1392 |
| **Smallest** | 0.5 | 0.4930 | 0.4271 | 0.3209 | 0.2954 | 0.3429 | 0.3421 | 0.1357 | 0.1656 |
|  |  |  |  |  |  |  |  |  |  |
| **AttnEntropy@12** | 1.0 | 0.5937 | 0.5025 | 0.3937 | 0.3460 | 0.4791 | 0.4448 | 0.2058 | 0.2173 |
| **AttnEntropy@12** | 0.7 | 0.6211 | 0.5228 | 0.4265 | 0.3711 | 0.5133 | 0.4647 | 0.2328 | 0.2351 |
| **AttnEntropy@12** | 0.5 | 0.6337 | 0.5349 | 0.4301 | 0.3787 | 0.5312 | 0.4743 | 0.2481 | 0.2439 |
|  |  |  |  |  |  |  |  |  |  |
| **ARS (regression)** | 1.0 | 0.6404 | 0.5302 | 0.4153 | 0.3652 | 0.5501 | 0.4826 | 0.2189 | 0.2249 |
| **ARS (regression)** | 0.7 | 0.6494 | 0.5470 | 0.4503 | 0.3920 | 0.5838 | 0.5037 | 0.2346 | 0.2357 |
| **ARS (regression)** | 0.5 | 0.6526 | 0.5523 | 0.4508 | 0.3909 | 0.5861 | 0.5064 | 0.2548 | 0.2486 |
|  |  |  |  |  |  |  |  |  |  |

## Method definitions (all reproducible via `<model>/baselines/run_baseline.sh <DS> <T> <method>` or the unified `experiments/run_all_baselines.sh`)

- **Greedy** — deterministic decoding (single response, temperature=0).
- **Pass@1** — response_1, bbox 1, no validity check. (Renamed from `first_response`.)
- **FirstValid** — first response with a parseable, coord-order-valid bbox.
- **TokEntropy** — candidate with lowest summed bbox-token vocab entropy.
- **MajVote** — per-coord majority vote across N=10 candidates → synthetic bbox (`bbox_aggregation_eval.py --agg basic`).
- **MeanBBox** — per-coord arithmetic mean across N=10 candidates → synthetic bbox (`bbox_aggregation_eval.py --agg mean`).
- **Smallest** — candidate with smallest area.
- **AttnEntropy@12** — 12 hand-picked (layer, head) attention entropy ranking (3 heads per coord, rank-sum).
- **ARS (regression)** — trained CNN regressor over attention maps; argmax predicted IoU per image.

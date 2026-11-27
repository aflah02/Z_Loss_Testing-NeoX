# Z-loss verification, 2026-11-27

The full matrix passed on two NVIDIA H100 PCIe GPUs in `gpt-neox:te` with DeepSpeed `0.16.5+0c91f204`. The run used the NeoX working tree based on `1d751e56` and recorded its [source diff](runs/20261127T074529Z-4S2JSW/megatron.diff).

| Check | Result |
| --- | --- |
| LM training | 24 configurations passed |
| Pairwise reward-model training | 11 configurations passed |
| Optimizer steps | 70 completed, 0 skipped, 0 NaN iterations |
| Evaluation | All 147 validation and test records reported base loss and its perplexity, with Z-loss metrics absent |
| Unsupported configurations | All 3 raised their designated errors |
| Numerical regression tests | All 7 passed |

At the first optimizer step, identical weights and tokens produced these metrics. The LM runs with omitted and explicit zero coefficients also matched through the second step, final validation, and test.

| Model | Z-loss coefficient | `train/lm_loss` | `train/z_loss` | `train/lm_loss_with_z_loss` |
| --- | ---: | ---: | ---: | ---: |
| FP32 LM | 0 | 6.223398 | — | — |
| FP32 LM | 0.0001 | 6.223398 | 39.038403 | 6.227302 |
| FP32 LM | 0.025 | 6.223398 | 39.038403 | 7.199358 |
| BF16 reward model | 0.025 | 0.72265625 | 0.02587890625 | 0.72265625 |
| BF16 reward model | 1.0 | 0.72265625 | 0.02587890625 | 0.75 |

The BF16 reward objective rounded to the base loss at the smaller coefficient. Increasing the coefficient changed the combined objective while the logged raw penalty stayed identical.

The [per-case summary](runs/20261127T074529Z-4S2JSW/summary.tsv) contains final training, validation, and test values. Each [case directory](runs/20261127T074529Z-4S2JSW/) includes the resolved YAML and trainer log. The [numerical test report](runs/20261127T074529Z-4S2JSW/z_loss_unit.log) includes the two-rank gradient and collective checks.

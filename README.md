# Z-loss training tests

The [verified H100 results](RESULTS.md) include the per-case report and first-step coefficient comparisons.

Run the full suite from the NeoX checkout with two CUDA GPUs, Docker with the NVIDIA runtime, and the local `gpt-neox:te` image.

```bash
bash logs/z_loss_testing/run_matrix.sh
```

List case names or select cases by name or shell pattern. A single-GPU case can run on either GPU.

```bash
bash logs/z_loss_testing/run_matrix.sh --list
GPU_DEVICES=1 bash logs/z_loss_testing/run_matrix.sh lm_fp32_on
bash logs/z_loss_testing/run_matrix.sh 'lm_*pp2*' 'rm_*'
```

`GPU_DEVICES` defaults to `0,1`. Set `GPT_NEOX_IMAGE` to use another NeoX development image with the branch's DeepSpeed fork. The runner uses the current checkout, generates small indexed datasets locally, and runs offline.

## Configuration coverage

The [matrix](configs/matrix.yml) pairs each profile with `z_loss: 0.0` and `z_loss: 0.025`. The [base config](configs/base.yml) runs two SGD steps on a two-layer transformer with two microbatches per update. The [LM data config](configs/lm.yml) masks every third target position. The [reward data config](configs/rm.yml) selects one chosen and one rejected score per example.

| Path | Cases |
| --- | --- |
| LM, standard CE | FP32, BF16, FP16 with FP32 CE, and native FP16 CE |
| LM, fused CE | FP32, BF16, and FP16 model outputs |
| LM parallelism | Single GPU, two tensor ranks, two data ranks, and one- and two-stage pipeline engines |
| Reward model | FP32, BF16, and FP16, plus FP32 tensor- and data-parallel runs |
| Coefficient controls | Omitted, explicit zero, `0.0001`, and `0.025`. An additional BF16 reward case uses `1.0` to make the contribution visible after rounding. |
| Validation and test | Periodic validation, final validation, and final test in every training case. Named and blended reporting also exercise validation before the first update. |
| Rejected configurations | Native FP16 CE with BF16, native FP16 CE with fused CE, and reward modeling with the pipeline engine must each raise their designated error. |

## Assertions and artifacts

The runner calls the same `train.main` used by NeoX training. On LM microbatches it gathers the vocabulary shards and independently computes masked cross entropy and squared log-sum-exp in FP64. On reward microbatches it checks the pairwise objective and the sum of the two selected squared scores.

Every optimizer step must complete and emit a finite `train/lm_loss`. With Z-loss active, the logger must also emit raw `train/z_loss` and the objective `train/lm_loss + coefficient * train/z_loss` under `train/lm_loss_with_z_loss`. The suite checks the reduced metrics against microbatch references averaged across the contributing ranks, allowing for the model's arithmetic precision. Validation and test use base loss and its perplexity. The checks inspect the actual CE coefficient passed during evaluation and the metric keys emitted by the trainer. First-step controls compare the raw metric at two coefficients using identical weights and inputs.

The full run also executes the branch's [Z-loss numerical regression tests](../../tests/model/test_z_loss.py). They compare gradients with a dense reference, count CE collectives, check saved vocabulary-shard tensors, and exercise long FP16 masks.

Each invocation saves a timestamped directory under `runs/`. Open `summary.tsv` for outcomes and final metrics. Elapsed seconds include process startup, dense reference calculations, and evaluation. Every case retains its resolved `config.yml` and trainer `run.log`. Successful training cases also save both updates and all validation and test events in `metrics.json`. Full runs save the numerical test log and JUnit XML alongside the summary. A failed case makes the runner exit nonzero.

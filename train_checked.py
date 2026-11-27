"""Run train.main and check the real microbatch losses and emitted scalars."""

import base64
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from megatron import logging, mpu, training
from megatron.model import gpt2_model
from train import main


def close(actual, expected, *, tolerance=2e-5):
    assert math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ), (float(actual), float(expected))


def run(config, output):
    coefficient = config.get("z_loss", 0.0)
    reward_model = config.get("train_impl") == "rm"
    phase = "train"
    emitted = {}
    eval_events = []
    train_references = []
    real_ce = training.cross_entropy
    real_forward = training.forward_step
    real_evaluate = training.evaluate
    real_train_log = training.training_log
    real_eval_log = training.log_eval_results
    real_emit = logging.tb_wandb_log
    loss_keys = (
        {"lm_loss", "z_loss", "lm_loss_with_z_loss"} if coefficient else {"lm_loss"}
    )
    reward_keys = (
        {"pos_values", "neg_values", "accuracy", "margin"} if reward_model else set()
    )

    def checked_ce(output, labels, **kwargs):
        active = coefficient if phase == "train" else 0.0
        assert kwargs["z_loss"] == active
        assert kwargs["_fused"] == config.get("cross_entropy_fusion", False)
        assert kwargs["_fp16"] == config.get("fp16_lm_cross_entropy", False)
        # Both production kernels may overwrite the input. Take the reference first.
        with torch.no_grad():
            group = mpu.get_model_parallel_group()
            shards = [
                torch.empty_like(output, memory_format=torch.contiguous_format)
                for _ in range(dist.get_world_size(group))
            ]
            dist.all_gather(shards, output.contiguous(), group=group)
            logits = torch.cat(shards, dim=-1).double()
            targets, mask = labels
            assert 0 < mask.count_nonzero() < mask.numel()
            ce = F.cross_entropy(
                logits.flatten(0, -2), targets.flatten(), reduction="none"
            )
            base = (ce * mask.flatten()).sum() / mask.double().sum()
            if active:
                raw = (logits.logsumexp(-1).square() * mask).sum() / mask.double().sum()
        loss = real_ce(output, labels, **kwargs)
        if active:
            metrics = kwargs["metrics"]
            close(metrics["lm_loss"], base)
            close(metrics["z_loss"], raw)
            close(loss, base + active * raw)
        else:
            tolerance = 0.002 if kwargs["_fp16"] else 2e-5
            close(loss, base, tolerance=tolerance)
        if phase == "train":
            train_references.append((float(base), float(raw) if active else 0.0))
        return loss

    def checked_forward(*args, **kwargs):
        result = real_forward(*args, **kwargs)
        if reward_model:
            loss, values = result
            active = coefficient if phase == "train" else 0.0
            # The fixture selects precisely one chosen and rejected score per microbatch.
            pos, neg = values["pos_values"], values["neg_values"]
            base = -F.logsigmoid(pos - neg)
            if active:
                raw = pos.square() + neg.square()
                close(values["lm_loss"], base)
                close(values["z_loss"], raw)
                close(loss, base + active * raw)
            else:
                assert set(values) == reward_keys
                close(loss, base)
            if phase == "train":
                train_references.append((float(base), float(raw) if active else 0.0))
        return result

    def checked_evaluate(*args, **kwargs):
        nonlocal phase
        phase = "eval"
        try:
            return real_evaluate(*args, **kwargs)
        finally:
            phase = "train"

    def checked_train_log(**kwargs):
        assert not kwargs["skipped_iter"]
        values = kwargs["loss_dict"]
        assert set(values) == loss_keys | reward_keys, values.keys()
        # Only the last pipeline stage observes the logits. Average its references
        # across microbatches and ranks, then check what the logger actually received.
        totals = torch.tensor(
            [
                sum(v[0] for v in train_references),
                sum(v[1] for v in train_references),
                len(train_references),
            ],
            dtype=torch.float64,
            device=values["lm_loss"].device,
        )
        dist.all_reduce(totals)
        train_references.clear()
        expected_count = (
            config["gradient_accumulation_steps"]
            * dist.get_world_size()
            // max(1, config["pipe_parallel_size"])
        )
        assert totals[2].item() == expected_count
        tolerance = (
            0.002 if config.get("fp16_lm_cross_entropy") and not coefficient else 2e-5
        )
        if reward_model:
            tolerance = {"fp32": 2e-5, "fp16": 0.001, "bfloat16": 0.008}[
                config["precision"]
            ]
        close(values["lm_loss"], totals[0] / expected_count, tolerance=tolerance)
        if coefficient:
            close(values["z_loss"], totals[1] / expected_count, tolerance=tolerance)
            close(
                values["lm_loss_with_z_loss"],
                values["lm_loss"] + coefficient * values["z_loss"],
                tolerance=tolerance,
            )
        return real_train_log(**kwargs)

    def capture_scalar(name, value, iteration, **kwargs):
        if name.startswith("train/") and "loss" in name and name != "train/loss_scale":
            emitted.setdefault(iteration, {})[name.removeprefix("train/")] = float(
                value
            )
        return real_emit(name, value, iteration, **kwargs)

    def checked_eval_log(**kwargs):
        values = kwargs["total_loss_dict"]
        assert set(values) == {"lm_loss", "lm_loss_ppl"} | reward_keys, values.keys()
        close(values["lm_loss_ppl"], math.exp(values["lm_loss"]))
        eval_events.append(
            dict(split=kwargs["chart_name"], prefix=kwargs["prefix"], **values)
        )
        return real_eval_log(**kwargs)

    with ExitStack() as stack:
        for module, name, replacement in (
            (training, "cross_entropy", checked_ce),
            (gpt2_model, "cross_entropy", checked_ce),
            (training, "forward_step", checked_forward),
            (training, "evaluate", checked_evaluate),
            (training, "training_log", checked_train_log),
            (training, "log_eval_results", checked_eval_log),
            (logging, "tb_wandb_log", capture_scalar),
        ):
            stack.enter_context(patch.object(module, name, replacement))
        encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
        main(input_args=["--megatron_config", encoded])
    if int(os.environ["RANK"]) == 0:
        steps = config["train_iters"]
        assert sorted(emitted) == list(range(1, steps + 1)), emitted
        for values in emitted.values():
            assert set(values) == loss_keys and all(
                math.isfinite(v) for v in values.values()
            )
        seen = {(event["split"], event["prefix"]) for event in eval_events}
        scheduled = set(
            range(config["eval_interval"], steps + 1, config["eval_interval"])
        )
        scheduled.update(config.get("extra_eval_iters", []))
        expected = set()
        for chart, split in (("validation", "valid"), ("test", "test")):
            charts = [chart]
            if config.get("eval_loss_logging", "blended") != "blended":
                charts = [f"{chart}/{name}" for name in config[f"{split}_data_names"]]
                if config["eval_loss_logging"] == "blended_and_separate":
                    charts.append(f"{chart}/blended")
            prefixes = [
                (
                    "the end of training for val data"
                    if split == "valid"
                    else "the end of training for test data"
                )
            ]
            if split == "valid":
                prefixes += [f"iteration {step}" for step in scheduled]
            expected.update((name, prefix) for name in charts for prefix in prefixes)
        assert seen == expected, (seen, expected)
        output.write_text(
            json.dumps({"train": emitted, "eval": eval_events}, indent=2) + "\n"
        )


if __name__ == "__main__":
    run(yaml.safe_load(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))

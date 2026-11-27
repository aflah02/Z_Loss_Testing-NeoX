"""Expand the YAML profiles, run each selected case, and retain its inputs and results."""

import argparse
import csv
import fnmatch
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


SUITE = Path(__file__).resolve().parent


def read_config(name):
    return yaml.safe_load((SUITE / "configs" / f"{name}.yml").read_text())


def cases():
    matrix = read_config("matrix")
    variants = {
        f"{name}_{suffix}": {**profile, "z_loss": coefficient}
        for name, profile in matrix["profiles"].items()
        for suffix, coefficient in matrix["coefficients"].items()
    }
    variants.update(matrix["additional"])
    rejected = matrix["rejected"]
    variants.update({name: entry["config"] for name, entry in rejected.items()})
    base, lm, rm = (read_config(name) for name in ("base", "lm", "rm"))
    for name, variant in variants.items():
        data = rm if variant.get("train_impl") == "rm" else lm
        cfg = {**base, **data, **variant}
        cfg["global_num_gpus"] = cfg["num_gpus"]
        if cfg["precision"] == "fp16":
            cfg["fp16"] = {"enabled": True, "loss_scale": 128}
        if cfg["precision"] == "bfloat16":
            cfg["data_types"] = {"grad_accum_dtype": "fp32"}
        yield name, cfg, rejected.get(name)


def check_coefficients(report):
    """The first update starts from identical weights and sees identical tokens."""

    def metrics(name):
        path = report / name / "metrics.json"
        return json.loads(path.read_text()) if path.is_file() else None

    default, off, small, on = (
        metrics(f"lm_fp32_{name}") for name in ("default", "off", "small", "on")
    )
    if default and off:
        assert default == off, "omitted and explicit zero Z-loss diverged"
    if small and on:
        for name in ("lm_loss", "z_loss"):
            assert math.isclose(
                small["train"]["1"][name], on["train"]["1"][name], rel_tol=1e-6
            ), name
        assert (
            small["train"]["1"]["lm_loss_with_z_loss"]
            < on["train"]["1"]["lm_loss_with_z_loss"]
        )
    rm_small, rm_large = metrics("rm_bf16_on"), metrics("rm_bf16_large")
    if rm_small and rm_large:
        small_step, large_step = rm_small["train"]["1"], rm_large["train"]["1"]
        assert small_step["z_loss"] == large_step["z_loss"]
        assert small_step["lm_loss"] == large_step["lm_loss"]
        assert large_step["lm_loss_with_z_loss"] > small_step["lm_loss_with_z_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns", nargs="*", help="shell-style case names (default: all)"
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [
        case
        for case in cases()
        if not args.patterns
        or any(fnmatch.fnmatchcase(case[0], pattern) for pattern in args.patterns)
    ]
    for pattern in args.patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name, _, _ in selected):
            parser.error(f"no cases match {pattern!r}")
    if args.list:
        print("\n".join(case[0] for case in selected))
        return

    import torch
    from prepare_data import prepare

    devices = torch.cuda.device_count()
    needed = max(config["num_gpus"] for _, config, _ in selected)
    if devices < needed:
        parser.error(f"selected cases require {needed} GPUs, found {devices}")
    prepare(Path("data"), read_config("base")["seq_length"])
    report = Path(os.environ["Z_LOSS_RESULTS"])
    failed = []
    with (report / "summary.tsv").open("w", newline="") as summary:
        table = csv.writer(summary, delimiter="\t", lineterminator="\n")
        table.writerow(
            [
                "case",
                "status",
                "seconds",
                "train_lm_loss",
                "train_z_loss",
                "train_lm_loss_with_z_loss",
                "validation_lm_loss",
                "test_lm_loss",
            ]
        )
        for name, config, expected_error in selected:
            directory = report / name
            directory.mkdir()
            config_path = directory / "config.yml"
            metrics = directory / "metrics.json"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [sys.executable]
            workers = config["num_gpus"]
            if workers > 1:
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    f"--nproc-per-node={workers}",
                    "--master-addr=127.0.0.1",
                    "--master-port=29654",
                    "--max-restarts=0",
                ]
            command += [str(SUITE / "train_checked.py"), str(config_path), str(metrics)]
            env = dict(
                os.environ,
                LOCAL_RANK="0",
                RANK="0",
                WORLD_SIZE=str(workers),
                LOCAL_WORLD_SIZE=str(workers),
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT="29654",
            )
            start = time.monotonic()
            print(f"Running {name}", flush=True)
            log = directory / "run.log"
            with log.open("w") as out:
                try:
                    with subprocess.Popen(
                        command,
                        env=env,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    ) as worker:
                        try:
                            code = worker.wait(timeout=240)
                        except subprocess.TimeoutExpired:
                            os.killpg(worker.pid, signal.SIGKILL)
                            code = worker.wait()
                            print("Timed out after 240 seconds", file=out)
                except OSError as error:
                    print(error, file=out)
                    code = -1
            seconds = round(time.monotonic() - start, 1)
            passed = code == 0 and metrics.is_file()
            values = [""] * 5
            if expected_error:
                exception = expected_error.get("exception", "ValueError")
                passed = code > 0 and any(
                    f"{exception}:" in line and expected_error["error"] in line
                    for line in log.read_text().splitlines()
                )
            elif passed:
                result = json.loads(metrics.read_text())
                last = result["train"][str(config["train_iters"])]
                # Named cases have one fixture per split, so its mean is also the blend.
                final = {
                    event["split"].split("/")[0]: event["lm_loss"]
                    for event in result["eval"]
                }
                values = [
                    last["lm_loss"],
                    last.get("z_loss", ""),
                    last.get("lm_loss_with_z_loss", ""),
                    final["validation"],
                    final["test"],
                ]
            status = (
                ("EXPECTED_REJECTION" if expected_error else "PASS")
                if passed
                else "FAIL"
            )
            table.writerow([name, status, seconds, *values])
            summary.flush()
            print(f"{status} {name} ({seconds}s)", flush=True)
            if not passed:
                failed.append(name)
                print("\n".join(log.read_text().splitlines()[-18:]), flush=True)
    if failed:
        sys.exit("Failed cases: " + ", ".join(failed))
    check_coefficients(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate missing commands for the pareto buffer-size grid.

MIN_RESULT_LINES is the exact expected scalar accuracy count (default: 500).
Valid short traces are incomplete; excess or malformed traces stop generation.
Values <= 0 select existence-only checking, not completeness validation.
Scalar structure does not prove round provenance.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Iterable

from utils.result_validation import IncompleteResultError, load_accuracy


OUTPUT_FILE = os.environ.get("OUTPUT_FILE", str(Path.home() / ".local/state/SAW-Implementation/queues/missing.sh"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(Path.home() / ".local/state/SAW-Implementation/runs/active/baselines"))
RESULT_DIRS = os.environ.get("RESULT_DIRS", str(Path(__file__).resolve().parents[1] / "experiment_results/pareto")).split()
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1 2 3 4 5").split()]
RANDOM_COST = os.environ.get("RANDOM_COST", "pareto")
MIN_RESULT_LINES = int(os.environ.get("MIN_RESULT_LINES", "500"))
ENV_SETUP = os.environ.get('ENV_SETUP', '')
PYTHON_BIN = os.environ.get('PYTHON_BIN', sys.executable)
PAIRS = [(40, 5), (80, 10), (60, 10)]

DATASET_ALPHAS = {
    "femnist": [0.5],
    "cifar": [0.5, 0.1],
    "cifar100": [0.5, 0.1, 0.01],
    "tinyimagenet": [0.5, 0.1, 0.01],
}

ALGOS = ['SAW', 'FedBuff', 'FedBuffMALight', 'FedAsync', 'CA2FL', 'CASA', 'FADAS', 'AdaMasFL']


def _format_alpha(alpha: float) -> str:
    return str(alpha)


def _expected_names(
    dataset: str,
    algo: str,
    alpha: float,
    random_cost: str,
    concurrency: int,
    buffer_size: int,
    seed: int,
) -> set[str]:
    alpha_text = _format_alpha(alpha)
    algo_name = algo
    names = {
        f"{dataset}-{algo_name}-{alpha_text}-{random_cost}-{concurrency}-{buffer_size}-{seed}-test_acc.txt",
    }
    if random_cost == "pareto":
        names.add(f"{dataset}-{algo_name}-{alpha_text}-{concurrency}-{buffer_size}-{seed}-test_acc.txt")
    return names


def _collect_existing_result_names(result_dirs: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for result_dir in result_dirs:
        root = Path(result_dir)
        if not root.exists():
            continue
        for path in root.rglob("*-test_acc.txt"):
            if MIN_RESULT_LINES > 0:
                try:
                    load_accuracy(path, expected_rounds=MIN_RESULT_LINES)
                except IncompleteResultError:
                    continue
            names.add(path.name)
    return names


def _dataset_flags(dataset: str) -> list[str]:
    flags: list[str] = []
    if dataset == "tinyimagenet":
        flags.extend(["--num_classes", "200"])
    return flags


def _quote_command(parts: list[str]) -> str:
    return shlex.join(parts)


def _build_command(
    dataset: str,
    algo: str,
    alpha: float,
    random_cost: str,
    concurrency: int,
    buffer_size: int,
    seed: int,
) -> str:
    parts = [
        PYTHON_BIN,
        str(Path(__file__).resolve().parent / "main_fed.py"),
        "--dataset",
        dataset,
        "--algo",
        algo,
        "--distribution",
        "noniid",
        "--alpha",
        _format_alpha(alpha),
        "--random_cost",
        random_cost,
        "--concurrency",
        str(concurrency),
        "--buffer_size",
        str(buffer_size),
        "--seed",
        str(seed),
        "--output_dir",
        str(Path(OUTPUT_DIR) / f'{dataset}-a{alpha}-Mc{concurrency}-b{buffer_size}'
            / algo / f'seed={seed}' / 'attempt=1'),
        "--gpu",
        "0",
    ]
    parts.extend(_dataset_flags(dataset))
    command = _quote_command(parts)
    if ENV_SETUP:
        command = f"source {shlex.quote(ENV_SETUP)} && {command}"
    return command


def main() -> None:
    existing_names = _collect_existing_result_names(RESULT_DIRS)
    total = 0
    missing: list[tuple[str, str, float, int, int, int]] = []

    for dataset, alphas in DATASET_ALPHAS.items():
        for alpha in alphas:
            for concurrency, buffer_size in PAIRS:
                for seed in SEEDS:
                    for algo in ALGOS:
                        total += 1
                        expected = _expected_names(
                            dataset,
                            algo,
                            alpha,
                            RANDOM_COST,
                            concurrency,
                            buffer_size,
                            seed,
                        )
                        if expected & existing_names:
                            continue
                        missing.append((dataset, algo, alpha, concurrency, buffer_size, seed))

    validation_mode = (
        f"exact_scalar_count expected_rounds={MIN_RESULT_LINES}; round provenance not audited"
        if MIN_RESULT_LINES > 0 else "existence_only; completeness not checked"
    )
    lines = [
        "# Auto-generated missing baseline tasks.",
        f"# random_cost={RANDOM_COST} seeds={SEEDS} pairs={PAIRS}",
        f"# min_result_lines={MIN_RESULT_LINES}",
        f"# validation_mode={validation_mode}",
        f"# env_setup={ENV_SETUP}",
        f"# python_bin={PYTHON_BIN}",
        f"# total={total} existing={total - len(missing)} missing={len(missing)}",
    ]
    for dataset, algo, alpha, concurrency, buffer_size, seed in missing:
        lines.append(
            _build_command(dataset, algo, alpha, RANDOM_COST, concurrency, buffer_size, seed)
        )

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"result dirs: {RESULT_DIRS}")
    print(f"validation mode: {validation_mode}")
    print(f"existing result files scanned: {len(existing_names)}")
    print(f"total tasks: {total}")
    print(f"existing tasks: {total - len(missing)}")
    print(f"missing tasks: {len(missing)}")
    print(f"wrote: {OUTPUT_FILE}")
    by_algo: dict[str, int] = {}
    by_case: dict[str, int] = {}
    for dataset, algo, alpha, concurrency, buffer_size, seed in missing:
        by_algo[algo] = by_algo.get(algo, 0) + 1
        key = f"{dataset}:alpha={alpha}:({concurrency},{buffer_size})"
        by_case[key] = by_case.get(key, 0) + 1
    print("missing by algorithm:")
    for algo in ALGOS:
        print(f"  {algo}: {by_algo.get(algo, 0)}")
    print("missing by dataset/config:")
    for key in sorted(by_case):
        print(f"  {key}: {by_case[key]}")


if __name__ == "__main__":
    main()

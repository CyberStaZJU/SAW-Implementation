#!/usr/bin/env python
"""Lightweight direction-skew logging for asynchronous FL runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.sampling import extract_labels

FIXED_MONITOR_GROUPS = 4


def _json_list(values: Iterable) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _num_classes(args, labels: np.ndarray) -> int:
    configured = int(getattr(args, "num_classes", 0) or 0)
    if labels.size == 0:
        return max(1, configured)
    observed = int(np.max(labels)) + 1
    return max(1, configured, observed)


def _default_num_groups(args, labels: np.ndarray) -> int:
    return FIXED_MONITOR_GROUPS


def _client_label_histograms(
    dataset_train,
    dict_users: dict,
    num_users: int,
    num_classes: int,
) -> np.ndarray:
    labels = extract_labels(dataset_train)
    histograms = np.zeros((num_users, num_classes), dtype=np.float64)
    for idx in range(num_users):
        raw_indices = dict_users[idx]
        if isinstance(raw_indices, set):
            raw_indices = sorted(raw_indices)
        user_indices = np.asarray(raw_indices, dtype=np.int64)
        if user_indices.size == 0:
            continue
        user_labels = labels[user_indices]
        hist = np.bincount(user_labels, minlength=num_classes).astype(np.float64)
        total = float(hist.sum())
        if total > 0:
            histograms[idx] = hist / total
    return histograms


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _balanced_targets(num_points: int, num_groups: int) -> np.ndarray:
    if num_points % num_groups != 0:
        raise ValueError(
            f"balanced direction-skew monitor requires num_users divisible by num_groups, "
            f"got num_users={num_points}, num_groups={num_groups}"
        )
    return np.full(num_groups, num_points // num_groups, dtype=np.int64)


def _balanced_spherical_assignment(features: np.ndarray, centroids: np.ndarray, targets: np.ndarray) -> np.ndarray:
    num_points, num_groups = features.shape[0], centroids.shape[0]
    sims = features @ centroids.T
    order = np.argsort(-sims, axis=None)
    assigned = np.full(num_points, -1, dtype=np.int64)
    remaining = targets.astype(np.int64, copy=True)

    for flat_idx in order:
        point_id = int(flat_idx // num_groups)
        group_id = int(flat_idx % num_groups)
        if assigned[point_id] >= 0 or remaining[group_id] <= 0:
            continue
        assigned[point_id] = group_id
        remaining[group_id] -= 1
        if np.all(assigned >= 0):
            break

    if np.any(assigned < 0):
        open_groups = [idx for idx, count in enumerate(remaining) for _ in range(int(count))]
        for point_id, group_id in zip(np.where(assigned < 0)[0], open_groups):
            assigned[int(point_id)] = int(group_id)

    return assigned


def _spherical_kmeans(features: np.ndarray, num_groups: int, seed: int, iters: int = 50) -> np.ndarray:
    """Balanced spherical K-means over normalized client label histograms."""
    num_points = int(features.shape[0])
    if num_points == 0:
        return np.array([], dtype=np.int64)

    num_groups = min(max(1, int(num_groups)), num_points)
    targets = _balanced_targets(num_points, num_groups)
    features = _l2_normalize(features.astype(np.float64, copy=False))
    rng = np.random.default_rng(seed)
    init = rng.choice(num_points, size=num_groups, replace=False)
    centroids = features[init].copy()
    assign = np.full(num_points, -1, dtype=np.int64)

    for _ in range(max(1, int(iters))):
        next_assign = _balanced_spherical_assignment(features, centroids, targets)
        if np.array_equal(assign, next_assign):
            break
        assign = next_assign

        for group_id in range(num_groups):
            members = np.where(assign == group_id)[0]
            centroid = features[members].mean(axis=0, keepdims=True)
            centroids[group_id] = _l2_normalize(centroid)[0]

    return assign


def _direction_skew_path(args, output_path: str) -> Path:
    log_dir = Path(getattr(args, "direction_skew_log_dir", ""))
    log_dir.mkdir(parents=True, exist_ok=True)

    output_name = Path(output_path).name
    if output_name.endswith("-test_acc.txt"):
        output_name = output_name[:-len("-test_acc.txt")] + "-direction_skew.csv"
    else:
        output_name = Path(output_name).stem + "-direction_skew.csv"
    return log_dir / output_name


def setup_direction_skew_monitor(args, state, dataset_train, dict_users, output_path: str) -> str | None:
    """Attach fixed monitor groups to state and initialize a CSV log.

    The monitor groups are algorithm-independent client clusters built from
    local label histograms. They are used only for measurement, not training.
    """

    log_dir = str(getattr(args, "direction_skew_log_dir", "") or "")
    if not log_dir:
        return None

    labels = extract_labels(dataset_train)
    num_users = int(getattr(args, "num_users", len(dict_users)))
    num_classes = _num_classes(args, labels)
    num_groups = _default_num_groups(args, labels)
    seed = int(getattr(args, "seed", 0))

    histograms = _client_label_histograms(dataset_train, dict_users, num_users, num_classes)
    group_ids = _spherical_kmeans(histograms, num_groups, seed=seed + 7919)
    group_sizes = np.bincount(group_ids, minlength=num_groups).astype(np.int64)
    monitor_mode = "balanced_label_histogram"

    state["direction_skew_monitor"] = {
        "num_groups": int(num_groups),
        "group_ids": group_ids.astype(np.int64).tolist(),
        "group_sizes": group_sizes.astype(np.int64).tolist(),
        "mode": monitor_mode,
    }

    path = _direction_skew_path(args, output_path)
    fieldnames = _direction_skew_fieldnames()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    print(f"Direction-skew log: {path}", flush=True)
    return str(path)


def _direction_skew_fieldnames() -> list[str]:
    return [
        "round",
        "algo",
        "dataset",
        "alpha",
        "random_cost",
        "concurrency",
        "buffer_size",
        "seed",
        "run_tag",
        "monitor_mode",
        "num_groups",
        "population_group_sizes",
        "buffer_count",
        "valid_count",
        "invalid_count",
        "buffer_counts",
        "valid_counts",
        "buffer_phi",
        "valid_phi",
        "buffer_coverage",
        "valid_coverage",
        "effective_coverage",
        "top1_share",
        "top2_share",
        "entropy",
        "cache_fill_total",
        "cache_only_groups",
        "underrepresented_groups",
        "regrouped",
    ]


def _summarize_clients(client_ids: list[int], monitor: dict) -> dict[str, object]:
    num_groups = int(monitor["num_groups"])
    group_ids = monitor["group_ids"]
    group_sizes = np.asarray(monitor["group_sizes"], dtype=np.float64)
    population = group_sizes / max(1.0, float(group_sizes.sum()))

    counts = np.zeros(num_groups, dtype=np.int64)
    for idx in client_ids:
        if 0 <= int(idx) < len(group_ids):
            group_id = int(group_ids[int(idx)])
            if 0 <= group_id < num_groups:
                counts[group_id] += 1

    total = int(counts.sum())
    if total <= 0:
        return {
            "counts": counts,
            "phi": "",
            "coverage": "",
            "top1_share": "",
            "top2_share": "",
            "entropy": "",
        }

    mixture = counts.astype(np.float64) / float(total)
    nonzero = mixture[mixture > 0]
    sorted_counts = np.sort(counts)[::-1]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / math.log(float(num_groups)) if num_groups > 1 else 0.0

    return {
        "counts": counts,
        "phi": float(np.sum((mixture - population) ** 2)),
        "coverage": float(np.count_nonzero(counts) / float(num_groups)),
        "top1_share": float(sorted_counts[0] / float(total)),
        "top2_share": float(sorted_counts[:2].sum() / float(total)),
        "entropy": entropy,
    }


def _format_float(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.8f}"


def _effective_coverage_from_state(state, fallback: str) -> tuple[str, float, int, int]:
    selected = state.get("last_selected_item_summary")
    if not selected:
        return fallback, 0.0, 0, 0

    num_groups = int(state.get("num_groups", 0) or 0)
    if num_groups <= 0:
        return fallback, 0.0, 0, 0

    represented = set()
    cache_fill_total = 0.0
    cache_only_groups = 0
    underrepresented_groups = 0
    for item in selected:
        group_id = int(item.get("group", -1))
        if group_id >= 0:
            represented.add(group_id)
        member_count = int(item.get("member_count", 0) or 0)
        cache_fill = float(item.get("cache_fill_count", 0.0) or 0.0)
        cache_fill_total += cache_fill
        if member_count == 0 and cache_fill > 0:
            cache_only_groups += 1
        elif member_count > 0 and cache_fill > 0:
            underrepresented_groups += 1

    return (
        f"{len(represented) / float(num_groups):.8f}",
        cache_fill_total,
        cache_only_groups,
        underrepresented_groups,
    )


def log_direction_skew_metrics(args, state, csv_path: str | None) -> None:
    if not csv_path:
        return

    monitor = state.get("direction_skew_monitor")
    if not monitor:
        return

    every = max(1, int(getattr(args, "direction_skew_log_every", 1) or 1))
    round_idx = int(state.get("iterations", 0))
    if round_idx % every != 0:
        return

    buffer_list = state.get("last_buffer_list", state.get("buffer_list", []))
    if not buffer_list:
        return

    buffer_list = [int(idx) for idx in buffer_list]
    valid_list = state.get("last_valid_buffer_list", buffer_list)
    invalid_list = state.get("last_invalid_buffer_list", [])
    valid_list = [int(idx) for idx in valid_list]
    invalid_list = [int(idx) for idx in invalid_list]

    buffer_summary = _summarize_clients(buffer_list, monitor)
    valid_summary = _summarize_clients(valid_list, monitor)
    effective_coverage, cache_fill_total, cache_only_groups, underrepresented_groups = (
        _effective_coverage_from_state(state, _format_float(buffer_summary["coverage"]))
    )

    row = {
        "round": round_idx,
        "algo": getattr(args, "algo", ""),
        "dataset": getattr(args, "dataset", ""),
        "alpha": "" if getattr(args, "distribution", "") in {"label_block", "label_correlated"} else getattr(args, "alpha", ""),
        "random_cost": getattr(args, "random_cost", ""),
        "concurrency": getattr(args, "concurrency", ""),
        "buffer_size": getattr(args, "buffer_size", ""),
        "seed": getattr(args, "seed", ""),
        "run_tag": getattr(args, "run_tag", ""),
        "monitor_mode": monitor.get("mode", ""),
        "num_groups": monitor["num_groups"],
        "population_group_sizes": _json_list(monitor["group_sizes"]),
        "buffer_count": len(buffer_list),
        "valid_count": len(valid_list),
        "invalid_count": len(invalid_list),
        "buffer_counts": _json_list(buffer_summary["counts"].astype(int).tolist()),
        "valid_counts": _json_list(valid_summary["counts"].astype(int).tolist()),
        "buffer_phi": _format_float(buffer_summary["phi"]),
        "valid_phi": _format_float(valid_summary["phi"]),
        "buffer_coverage": _format_float(buffer_summary["coverage"]),
        "valid_coverage": _format_float(valid_summary["coverage"]),
        "effective_coverage": effective_coverage,
        "top1_share": _format_float(buffer_summary["top1_share"]),
        "top2_share": _format_float(buffer_summary["top2_share"]),
        "entropy": _format_float(buffer_summary["entropy"]),
        "cache_fill_total": f"{cache_fill_total:.8f}",
        "cache_only_groups": cache_only_groups,
        "underrepresented_groups": underrepresented_groups,
        "regrouped": int(bool(state.get("last_regrouped", False))),
    }

    with Path(csv_path).open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_direction_skew_fieldnames())
        writer.writerow(row)

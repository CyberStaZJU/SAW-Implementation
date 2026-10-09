#!/usr/bin/env python3
"""Build a truncated FEMNIST directory for faster experiments."""

import argparse
import json
import os
from pathlib import Path


def _json_files(root, split):
    split_dir = root / split
    return sorted(path for path in split_dir.iterdir() if path.suffix == ".json")


def _load_user_data(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("user_data", {})


def _count_train_clients(root):
    counts = {}
    for path in _json_files(root, "train"):
        for user_id, user_data in _load_user_data(path).items():
            counts[user_id] = len(user_data.get("x", []))
    return counts


def _select_clients(counts, num_users, min_samples):
    eligible = sorted(
        user_id for user_id, count in counts.items()
        if count >= min_samples
    )
    if num_users > len(eligible):
        raise SystemExit(
            f"Requested {num_users} clients, but only {len(eligible)} clients "
            f"have at least {min_samples} train samples."
        )
    return eligible[:num_users]


def _collect_split(root, split, selected_client_ids):
    selected = set(selected_client_ids)
    collected = {}
    for path in _json_files(root, split):
        for user_id, user_data in _load_user_data(path).items():
            if user_id in selected:
                collected[user_id] = user_data
    return collected


def _write_split(path, user_data, selected_client_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_users = [user_id for user_id in selected_client_ids if user_id in user_data]
    payload = {
        "users": ordered_users,
        "num_samples": [len(user_data[user_id].get("x", [])) for user_id in ordered_users],
        "user_data": {user_id: user_data[user_id] for user_id in ordered_users},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _sample_count(user_data):
    return sum(len(payload.get("x", [])) for payload in user_data.values())


def main():
    parser = argparse.ArgumentParser(
        description="Create an external FEMNIST subset from full LEAF/FEMNIST JSON files."
    )
    data_root = Path(os.environ.get("SAW_DATA_ROOT", str(Path.home() / "SAW-data")))
    parser.add_argument("--root", type=Path, default=data_root / "femnist")
    parser.add_argument("--output", type=Path, default=data_root / "femnist_truncated")
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--min-samples-per-client", type=int, default=250)
    args = parser.parse_args()
    for path in (args.root, args.output):
        if path is not None and path.expanduser().resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error(f"Preparation path must be outside the repository: {path}")

    counts = _count_train_clients(args.root)
    selected_client_ids = _select_clients(
        counts,
        args.num_users,
        args.min_samples_per_client,
    )
    selected_counts = [counts[user_id] for user_id in selected_client_ids]

    train_user_data = _collect_split(args.root, "train", selected_client_ids)
    test_user_data = _collect_split(args.root, "test", selected_client_ids)

    _write_split(args.output / "train" / "femnist_truncated_train.json", train_user_data, selected_client_ids)
    _write_split(args.output / "test" / "femnist_truncated_test.json", test_user_data, selected_client_ids)

    print(
        f"eligible clients: {sum(1 for count in counts.values() if count >= args.min_samples_per_client)} / {len(counts)}"
    )
    print(f"selected clients: {len(selected_client_ids)}")
    print(
        f"selected train samples: {sum(selected_counts)} "
        f"(min={min(selected_counts)}, max={max(selected_counts)})"
    )
    print(f"written train samples: {_sample_count(train_user_data)}")
    print(f"written test samples : {_sample_count(test_user_data)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

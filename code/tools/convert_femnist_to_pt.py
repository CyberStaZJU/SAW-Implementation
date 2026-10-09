#!/usr/bin/env python3
"""
Convert FEMNIST truncated JSON files to PyTorch .pt binary format.

This dramatically speeds up data loading by eliminating JSON parsing
(~1.6GB JSON → ~240MB .pt, load time from ~60s to ~1s).

Output format per split (train/test):
    femnist_truncated_{split}.pt
        ├── images:       FloatTensor, shape (N, 784)
        ├── targets:      LongTensor,  shape (N,)
        ├── clients:      LongTensor,  shape (N,)  — client index per sample
        ├── client_ids:   list[str]    — ordered client ID strings
        └── num_samples:  list[int]    — samples per client (same order as client_ids)

Usage:
    python convert_femnist_to_pt.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]

Defaults:
    data_dir   = $SAW_DATA_ROOT/femnist_truncated (default: ~/SAW-data/femnist_truncated)
    output_dir = same as data_dir  (.pt files sit alongside JSON)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def convert_split(json_path, output_path):
    """Convert one JSON split (train or test) to a single .pt file."""
    print(f"  Loading JSON: {json_path}")
    t0 = time.time()
    with open(json_path, "r") as f:
        raw = json.load(f)
    t_json = time.time() - t0
    print(f"    JSON parse: {t_json:.1f}s")

    users = raw["users"]
    num_samples = raw["num_samples"]
    user_data = raw["user_data"]

    all_images = []
    all_targets = []
    all_clients = []

    for client_idx, uid in enumerate(users):
        ud = user_data[uid]
        n = len(ud["x"])
        all_images.extend(ud["x"])       # each is a list of 784 floats
        all_targets.extend(ud["y"])      # each is an int
        all_clients.extend([client_idx] * n)

    t1 = time.time()

    images_tensor = torch.tensor(all_images, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.int64)
    clients_tensor = torch.tensor(all_clients, dtype=torch.int64)

    data_dict = {
        "images": images_tensor,
        "targets": targets_tensor,
        "clients": clients_tensor,
        "client_ids": users,
        "num_samples": num_samples,
    }

    print(f"    Tensor conversion: {time.time() - t1:.1f}s")
    print(f"    images:    {images_tensor.shape}  ({images_tensor.element_size()*images_tensor.numel()/1e6:.1f} MB)")
    print(f"    targets:   {targets_tensor.shape}")
    print(f"    clients:   {clients_tensor.shape}")
    print(f"    #clients:  {len(users)}, #samples: {len(all_targets)}")

    t2 = time.time()
    torch.save(data_dict, output_path)
    t_save = time.time() - t2
    pt_size = os.path.getsize(output_path) / 1e6
    print(f"    Saved .pt: {output_path}  ({pt_size:.1f} MB, save time: {t_save:.1f}s)")


def main():
    default_data_dir = os.path.join(
        os.environ.get("SAW_DATA_ROOT", os.path.expanduser("~/SAW-data")), "femnist_truncated"
    )

    parser = argparse.ArgumentParser(description="Convert FEMNIST JSON to .pt")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Root directory containing train/ and test/ subdirs with JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for .pt files (default: same as data_dir)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_dir
    for path in (data_dir, output_dir):
        if Path(path).resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error(f"Preparation path must be outside the repository: {path}")

    for split in ("train", "test"):
        json_name = f"femnist_truncated_{split}.json"
        pt_name = f"femnist_truncated_{split}.pt"

        json_path = os.path.join(data_dir, split, json_name)
        pt_path = os.path.join(output_dir, split, pt_name)

        if not os.path.isfile(json_path):
            print(f"  SKIP {split}: {json_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Converting {split} split")
        print(f"{'='*60}")

        os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        convert_split(json_path, pt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

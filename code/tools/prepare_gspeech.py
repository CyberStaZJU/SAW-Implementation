#!/usr/bin/env python3
"""Download and prepare Google Speech Commands v2 for GSpeech."""

import argparse
import json
import os
import tarfile
import urllib.request
from pathlib import Path


DOWNLOAD_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
BACKGROUND_DIR = "_background_noise_"


def _download(url, archive_path):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        print(f"Archive exists: {archive_path}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)
    print(f"Saved: {archive_path}")


def _extract(archive_path, raw_dir):
    marker = raw_dir / "testing_list.txt"
    if marker.exists():
        print(f"Raw data exists: {raw_dir}")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {raw_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(raw_dir)


def _read_split_file(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _speaker_id(relative_path):
    name = Path(relative_path).name
    return name.split("_nohash_", 1)[0]


def _append_user_sample(user_data, client_id, x_value, y_value):
    payload = user_data.setdefault(client_id, {"x": [], "y": []})
    payload["x"].append(x_value)
    payload["y"].append(y_value)


def _build_splits(raw_dir, include_validation_in_train=False):
    validation = _read_split_file(raw_dir / "validation_list.txt")
    testing = _read_split_file(raw_dir / "testing_list.txt")
    train_user_data = {}
    test_user_data = {}
    labels = []

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name == BACKGROUND_DIR:
            continue
        label = label_dir.name
        labels.append(label)
        for wav_path in sorted(label_dir.glob("*.wav")):
            relative = f"{label}/{wav_path.name}"
            x_value = f"raw/{relative}"
            client_id = _speaker_id(relative)
            if relative in testing:
                _append_user_sample(test_user_data, client_id, x_value, label)
            elif relative not in validation or include_validation_in_train:
                _append_user_sample(train_user_data, client_id, x_value, label)

    return labels, train_user_data, test_user_data


def _write_json(path, user_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "users": sorted(user_data),
        "num_samples": [len(user_data[user]["x"]) for user in sorted(user_data)],
        "user_data": {user: user_data[user] for user in sorted(user_data)},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _count_samples(user_data):
    return sum(len(payload["x"]) for payload in user_data.values())


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Google Speech Commands v2 as LEAF-style GSpeech JSON."
    )
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("SAW_DATA_ROOT", str(Path.home() / "SAW-data"))) / "gspeech")
    parser.add_argument("--url", default=DOWNLOAD_URL)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--include-validation-in-train",
        action="store_true",
        help="fold validation_list.txt samples into train instead of skipping them",
    )
    args = parser.parse_args()
    for path in (args.root, args.archive):
        if path is not None and path.expanduser().resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error(f"Preparation path must be outside the repository: {path}")

    root = args.root
    raw_dir = root / "raw"
    archive_path = args.archive or root / "speech_commands_v0.02.tar.gz"

    if args.download:
        _download(args.url, archive_path)
    if args.extract or args.download:
        _extract(archive_path, raw_dir)

    if not (raw_dir / "testing_list.txt").exists():
        raise SystemExit(
            f"Missing raw GSpeech data under {raw_dir}. "
            "Run with --download, or put the extracted Speech Commands v0.02 files there."
        )

    labels, train_user_data, test_user_data = _build_splits(
        raw_dir,
        include_validation_in_train=args.include_validation_in_train,
    )
    _write_json(root / "train" / "gspeech_train.json", train_user_data)
    _write_json(root / "test" / "gspeech_test.json", test_user_data)

    print(f"labels: {len(labels)}")
    print(f"train clients: {len(train_user_data)}, samples: {_count_samples(train_user_data)}")
    print(f"test clients : {len(test_user_data)}, samples: {_count_samples(test_user_data)}")
    print(f"wrote: {root / 'train' / 'gspeech_train.json'}")
    print(f"wrote: {root / 'test' / 'gspeech_test.json'}")


if __name__ == "__main__":
    main()

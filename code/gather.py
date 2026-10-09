#!/usr/bin/env python3
"""Preview result families; --copy preserves sources under awaiting_review.

Keep paths relative to the source root, including attempt/accuracy/metrics.
Require paired accuracy, system metrics, resolved configuration and identity.
This is a structural collection check, not an exit-status or formal audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from utils.result_validation import load_accuracy

MIN_LINES = 500
STATE_ROOT = Path(os.environ.get('SAW_STATE_ROOT', str(Path.home() / '.local/state/SAW-Implementation')))
INPUT_DIR = os.environ.get('SAW_RAW_ROOT', str(STATE_ROOT / 'runs/active'))
OUTPUT_DIR = os.environ.get('SAW_GATHER_ROOT', str(STATE_ROOT / 'runs/awaiting_review'))
ACCURACY_SUFFIX = '-test_acc.txt'
IDENTITY_FIELDS = ('algo', 'dataset', 'seed', 'num_users', 'concurrency', 'buffer_size',
                   'random_cost', 'total_rounds')
METRIC_NUMBERS = ('runtime_sec', 'round_runtime_sec', 'simulated_wall_time',
                  'process_memory_mb', 'process_peak_memory_mb', 'device_peak_memory_mb')


@dataclass(frozen=True)
class CopyItem:
    source: Path
    destination: Path
    signature: tuple[int, int, int, int]


def signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def read_object(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def unique_file(candidates: list[Path], label: str) -> Path:
    found = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(found) != 1:
        raise ValueError(f'{label}: expected one paired file, found {len(found)}: {candidates}')
    return found[0]


def attempt_root(accuracy: Path) -> Path:
    return accuracy.parent.parent if accuracy.parent.name == 'accuracy' else accuracy.parent


def result_family(accuracy: Path) -> tuple[list[Path], Path, Path, Path]:
    root = attempt_root(accuracy)
    prefix = accuracy.name.removesuffix(ACCURACY_SUFFIX)
    resolved = unique_file([
        accuracy.with_name(accuracy.stem + '-resolved_config.json'),
        accuracy.parent / 'resolved_config.json',
    ], 'resolved config')
    metrics = unique_file([
        accuracy.parent / (prefix + '-system_metrics.csv'),
        root / 'metrics' / (prefix + '-system_metrics.csv'),
    ], 'system metrics')
    identity_candidates = list(dict.fromkeys(path for path in [
        root / 'identity.json',
        accuracy.with_name(accuracy.stem + '-identity.json'),
        accuracy.with_name(prefix + '-identity.json'),
    ] if path.is_file()))
    trainer_identity = accuracy.with_name(accuracy.stem + '-identity.json')
    scheduler_identity = root / 'identity.json'
    if (len(identity_candidates) == 2 and trainer_identity in identity_candidates
            and scheduler_identity in identity_candidates
            and read_object(trainer_identity).get('producer') == 'main_fed'):
        identity = trainer_identity
    else:
        identity = unique_file(identity_candidates, 'identity')
    if (resolved.name == 'resolved_config.json' or scheduler_identity in identity_candidates):
        if len(list(accuracy.parent.glob('*' + ACCURACY_SUFFIX))) != 1:
            raise ValueError(f'{root}: generic config/identity is ambiguous across multiple accuracies')
    members = {accuracy, resolved, metrics, *identity_candidates}
    for directory in {accuracy.parent, metrics.parent}:
        members.update(path for path in directory.glob(prefix + '-*')
                       if path.is_file() and path.suffix in {'.txt', '.csv', '.json', '.log'})
    for name in ('completion.json', 'audit.json', 'exit_code.txt', 'train.log', 'trainer.log'):
        path = root / name
        if path.is_file():
            members.add(path)
    return sorted(members), resolved, metrics, identity


def validate_identity(identity: dict, resolved: dict, accuracy: Path, total_rounds: int) -> dict:
    actual = resolved.get('args')
    if not isinstance(actual, dict) or any(field not in actual for field in IDENTITY_FIELDS):
        raise ValueError(f'{accuracy}: resolved args lack run identity fields')
    if actual['total_rounds'] != total_rounds:
        raise ValueError(f'{accuracy}: configured total_rounds differs from requested {total_rounds}')
    commit = resolved.get('source_commit')
    code = identity.get('code_identity', {})
    if not isinstance(code, dict):
        raise ValueError(f'{accuracy}: invalid code_identity')
    if (not isinstance(commit, str) or commit in ('', 'unrecorded', 'unknown')
            or code.get('commit', identity.get('source_commit')) != commit):
        raise ValueError(f'{accuracy}: missing or mismatched source commit')
    if 'source_manifest' in identity and identity['source_manifest'] != resolved.get('source_manifest'):
        raise ValueError(f'{accuracy}: mismatched source manifest')
    if isinstance(identity.get('args'), dict):
        expected = {field: identity['args'].get(field) for field in IDENTITY_FIELDS}
        expected.update({field: value for field, value in identity['args'].items()
                         if field in actual})
    else:
        task, config = identity.get('task', {}), identity.get('config', {})
        if not isinstance(task, dict) or not isinstance(config, dict):
            raise ValueError(f'{accuracy}: invalid task/config identity')
        expected = {'algo': identity.get('method'), 'dataset': task.get('dataset'),
                    'seed': identity.get('seed'), 'num_users': config.get('num_users'),
                    'concurrency': config.get('concurrency'), 'buffer_size': config.get('buffer')}
        if task.get('alpha') is not None:
            expected['alpha'] = task['alpha']
    if any(actual.get(field) != value for field, value in expected.items()):
        raise ValueError(f'{accuracy}: resolved args differ from attempt identity')
    suffix = '-'.join(str(actual[field]) for field in ('random_cost', 'concurrency', 'buffer_size', 'seed'))
    if actual.get('run_tag'):
        suffix += '-' + str(actual['run_tag'])
    if (not accuracy.name.startswith(f"{actual['dataset']}-{actual['algo']}-")
            or not accuracy.name.endswith('-' + suffix + ACCURACY_SUFFIX)):
        raise ValueError(f'{accuracy}: filename differs from resolved run identity')
    partition = accuracy.name[len(f"{actual['dataset']}-{actual['algo']}-"):-len('-' + suffix + ACCURACY_SUFFIX)]
    if actual.get('distribution', 'noniid') == 'noniid':
        if actual.get('alpha') is None or float(partition) != float(actual['alpha']):
            raise ValueError(f'{accuracy}: filename alpha differs from resolved args')
    elif actual['distribution'] in ('iid', 'label_block') and partition != actual['distribution']:
        raise ValueError(f'{accuracy}: filename partition differs from resolved args')
    command = identity.get('command')
    if command is not None:
        if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
            raise ValueError(f'{accuracy}: command must be a token list')
        for index, flag in enumerate(command):
            if not flag.startswith('--'):
                continue
            field = flag[2:]
            if field not in actual:
                continue
            expected_value = actual[field]
            if isinstance(expected_value, bool):
                matches = expected_value
            elif index + 1 >= len(command):
                matches = False
            else:
                provided = command[index + 1]
                try:
                    if isinstance(expected_value, int):
                        matches = int(provided) == expected_value
                    elif isinstance(expected_value, float):
                        matches = float(provided) == expected_value
                    else:
                        matches = provided == str(expected_value)
                except ValueError:
                    matches = False
            if not matches:
                raise ValueError(f'{accuracy}: command/config mismatch for {field}')
    elif not isinstance(identity.get('args'), dict):
        raise ValueError(f'{accuracy}: scheduler identity lacks command')
    for part in accuracy.parts:
        if part.startswith('attempt=') and str(identity.get('attempt')) != part.split('=', 1)[1]:
            raise ValueError(f'{accuracy}: attempt directory differs from identity')
        if part.startswith('seed=') and str(actual['seed']) != part.split('=', 1)[1]:
            raise ValueError(f'{accuracy}: seed directory differs from resolved args')
    return actual


def validate_family(accuracy: Path, resolved_path: Path, metrics_path: Path,
                    identity_path: Path, total_rounds: int) -> None:
    resolved, identity = read_object(resolved_path), read_object(identity_path)
    actual = validate_identity(identity, resolved, accuracy, total_rounds)
    scheduler_identity = attempt_root(accuracy) / 'identity.json'
    if scheduler_identity.is_file() and scheduler_identity != identity_path:
        scheduler = read_object(scheduler_identity)
        validate_identity(scheduler, resolved, accuracy, total_rounds)
        for record in (identity, scheduler):
            if 'cache_device' in record and record['cache_device'] != resolved.get('cache_device'):
                raise ValueError(f'{accuracy}: cache device differs from resolved configuration')
    interval = actual.get('interval', 1)
    if not isinstance(interval, int) or isinstance(interval, bool) or not 0 < interval <= total_rounds:
        raise ValueError(f'{accuracy}: invalid evaluation interval')
    load_accuracy(accuracy, expected_rounds=total_rounds // interval)
    with metrics_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {'round', 'algo', 'dataset', 'seed', 'concurrency', 'buffer_size',
                    'random_cost', 'alpha', 'run_tag', *METRIC_NUMBERS}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f'{metrics_path}: missing or duplicate metric columns')
        rows = list(reader)
    if len(rows) != total_rounds:
        raise ValueError(f'{metrics_path}: expected {total_rounds} metric rounds, found {len(rows)}')
    for number, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'{metrics_path}: malformed row {number}')
        if int(row['round']) != number:
            raise ValueError(f'{metrics_path}: non-contiguous round at row {number}')
        for field in ('algo', 'dataset', 'random_cost', 'run_tag'):
            if row[field] != str(actual.get(field, '')):
                raise ValueError(f'{metrics_path}: {field} differs from resolved args at round {number}')
        for field in ('seed', 'concurrency', 'buffer_size'):
            if int(row[field]) != actual[field]:
                raise ValueError(f'{metrics_path}: {field} differs from resolved args at round {number}')
        if actual.get('alpha') is not None and float(row['alpha']) != float(actual['alpha']):
            raise ValueError(f'{metrics_path}: alpha differs from resolved args at round {number}')
        for field in METRIC_NUMBERS:
            if field == "device_peak_memory_mb" and row[field] == "":
                continue
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{metrics_path}: invalid {field} at round {number}')


def review_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if 'verified' in path.parts:
        raise ValueError('Collection cannot target verified results; use awaiting_review')
    if 'awaiting_review' not in path.parts:
        path /= 'awaiting_review'
    repository = Path(__file__).resolve().parents[1]
    if path.is_relative_to(repository):
        raise ValueError(f'Collection output must be outside the source repository: {path}')
    return path


def preflight(sources: list[Path], target_root: Path, total_rounds: int) -> list[CopyItem]:
    if not isinstance(total_rounds, int) or isinstance(total_rounds, bool) or total_rounds <= 0:
        raise ValueError('total_rounds must be a positive integer')
    plan: dict[Path, CopyItem] = {}
    for source in sources:
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            if not source.name.endswith(ACCURACY_SUFFIX):
                raise ValueError(f'{source}: file source must be a test accuracy anchor')
            accuracies = [source]
            root = attempt_root(source)
            scope = root.parent if root.name.startswith('attempt=') else root
        else:
            accuracies = sorted(source.rglob('*' + ACCURACY_SUFFIX))
            scope = source
        if target_root.is_relative_to(scope) or scope.is_relative_to(target_root):
            raise ValueError(f'Source and collection roots must not overlap: {scope}, {target_root}')
        for accuracy in accuracies:
            members, resolved, metrics, identity = result_family(accuracy)
            snapshots = {}
            for member in members:
                if not member.is_relative_to(scope):
                    raise ValueError(f'{member}: source root must contain the entire attempt')
                if any(path.is_symlink() for path in (member, *member.parents) if path != scope.parent):
                    raise ValueError(f'{member}: symlinked evidence is not supported')
                snapshots[member] = signature(member)
            validate_family(accuracy, resolved, metrics, identity, total_rounds)
            for member in members:
                if signature(member) != snapshots[member]:
                    raise ValueError(f'{member}: evidence changed during preflight')
                destination = target_root / member.relative_to(scope)
                if destination in plan and plan[destination].source != member:
                    raise FileExistsError(f'Multiple sources map to {destination}')
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                for parent in destination.parents:
                    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                        raise FileExistsError(f'Destination parent is not a plain directory: {parent}')
                plan[destination] = CopyItem(member, destination, snapshots[member])
    return list(plan.values())


def copy_plan(plan: list[CopyItem]) -> None:
    created_files, created_dirs = [], []
    try:
        for item in plan:
            if signature(item.source) != item.signature:
                raise ValueError(f'{item.source}: evidence changed after preflight')
        for item in plan:
            missing = []
            parent = item.destination.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                created_dirs.append(directory)
            with item.destination.open('xb') as output:
                created_files.append(item.destination)
                with item.source.open('rb') as source:
                    shutil.copyfileobj(source, output)
            shutil.copystat(item.source, item.destination)
        for item in plan:
            if signature(item.source) != item.signature:
                raise ValueError(f'{item.source}: evidence changed while copying')
    except Exception:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


def collect_results(sources: list[Path], output_dir: Path, total_rounds: int = MIN_LINES,
                    apply: bool = False) -> int:
    """Preflight the whole batch before writing; never move or certify results."""
    target = review_root(output_dir)
    plan = preflight(sources, target, total_rounds)
    action = 'copy' if apply else 'dry-run'
    for item in plan:
        print(f'{action} awaiting_review: {item.source} -> {item.destination}')
    if apply:
        copy_plan(plan)
    print(f'status=awaiting_review mode={action} files={len(plan)}; formal audit still required')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--input-dir', default=INPUT_DIR,
                        help='recursive source root containing complete attempts')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help='external review root; awaiting_review is appended if absent')
    parser.add_argument('--total-rounds', '--min-lines', type=int, default=MIN_LINES,
                        help='required configured final round (default: 500)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--copy', action='store_true', help='copy families; preserve all originals')
    mode.add_argument('--dry-run', action='store_true', help='preview only (the default)')
    args = parser.parse_args(argv)
    try:
        return collect_results([Path(args.source)], Path(args.output_dir), args.total_rounds, args.copy)
    except (OSError, ValueError, TypeError) as exc:
        print(f'reject batch: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

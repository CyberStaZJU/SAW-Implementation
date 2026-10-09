#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import gc
import csv
import json
import math
import os
import random
import time
import tracemalloc
import resource
import subprocess
import sys
from pathlib import Path

from torch.utils.data import Subset

import matplotlib as mpl
import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass

from algorithm.dispatcher import init_state, run_one_round
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from models.test import test_img
from utils.direction_skew_logging import (
    log_direction_skew_metrics,
    setup_direction_skew_monitor,
)
from utils.options import args_parser
from utils.state_dict_ops import resolve_cache_device

mpl.use('Agg')
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def _fraction_token(value):
    text = f"{float(value):.3g}"
    return text.replace('.', 'p')


def npu_is_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def external_path(path, label):
    resolved = Path(path).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if resolved.is_relative_to(source_root):
        raise ValueError(f'{label} must be outside the source repository: {resolved}')
    return str(resolved)


def configure_runtime(args):
    args.data_dir = external_path(
        getattr(args, 'data_dir', '') or os.environ.get('SAW_DATA_ROOT', str(Path.home() / 'SAW-data')),
        'data_dir')
    if getattr(args, 'direction_skew_log_dir', ''):
        args.direction_skew_log_dir = external_path(args.direction_skew_log_dir, 'direction_skew_log_dir')
    if args.gpu == '-1':
        args.device = 'cpu'
    else:
        if npu_is_available():
            os.environ['ASCEND_DEVICE_ID'] = str(args.gpu)
            args.device = f'npu:{args.gpu}'
            torch.npu.set_device(args.device)
        elif torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if npu_is_available():
        torch.npu.manual_seed_all(args.seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def build_training_state(args):
    dataset_train, dataset_test, dict_users, num_samples, _ = build_dataset(args)
    validation_dataset = None
    validation_fraction = float(getattr(args, 'validation_fraction', 0.0) or 0.0)
    if validation_fraction > 0.0:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError('validation_fraction must be between 0 and 1')
        split_rng = np.random.default_rng(int(getattr(args, 'validation_split_seed', 731)))
        indices = split_rng.permutation(len(dataset_train))
        val_size = max(1, int(round(len(indices) * validation_fraction)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        validation_base = copy.deepcopy(dataset_train)
        if hasattr(validation_base, 'transform') and hasattr(dataset_test, 'transform'):
            validation_base.transform = dataset_test.transform
        validation_dataset = Subset(validation_base, val_indices.tolist())
        dataset_train = Subset(dataset_train, train_indices.tolist())
        remapped_users = {}
        train_lookup = {int(old): int(new) for new, old in enumerate(train_indices.tolist())}
        for client_id, client_indices in dict_users.items():
            remapped_users[client_id] = np.asarray(
                [train_lookup[int(index)] for index in client_indices if int(index) in train_lookup],
                dtype='int64',
            )
        dict_users = remapped_users
        num_samples = np.asarray([len(dict_users[i]) for i in range(args.num_users)])
        args.validation_size = int(len(validation_dataset))
        args.training_size_after_validation_split = int(len(dataset_train))
    net_glob, w_glob = build_model(args)
    state = {
        'net_glob': net_glob,
        'w_glob': w_glob,
        'dataset_train': dataset_train,
        'dict_users': dict_users,
        'num_samples': num_samples,
        'global_cost': 0,
    }
    return init_state(state, args), dataset_train, dataset_test, dict_users, num_samples, validation_dataset


def build_testacc_filename(args):
    parts = [args.dataset, args.algo]
    random_cost_name = getattr(args, 'random_cost', 'pareto')
    append_random_cost = True

    if args.distribution == 'iid':
        parts.append(args.distribution)
    elif args.distribution == 'label_block':
        parts.append(args.distribution)
    elif args.distribution == 'label_correlated':
        parts.append(f"{args.distribution}_p{_fraction_token(getattr(args, 'label_primary_fraction', 0.5))}")
    elif args.distribution == 'speed_label_correlated_dirichlet':
        parts.append(
            f"{args.distribution}_p{_fraction_token(getattr(args, 'label_primary_fraction', 0.9))}"
            f"_alpha{_fraction_token(getattr(args, 'alpha', 0.1))}"
        )
    else:
        parts.append(str(args.alpha))

    if append_random_cost:
        parts.append(random_cost_name)
    parts.extend([str(args.concurrency), str(args.buffer_size), str(args.seed)])
    if getattr(args, 'run_tag', ''):
        parts.append(str(args.run_tag))
    return '-'.join(parts) + '-test_acc.txt'


def build_output_path(args):
    filename = build_testacc_filename(args)
    output_dir = getattr(args, 'output_dir', '')
    if output_dir:
        output_dir = external_path(output_dir, 'output_dir')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    state_root = Path(os.environ.get('SAW_STATE_ROOT', str(Path.home() / '.local/state/SAW-Implementation')))
    setting = f'{args.dataset}-a{args.alpha}-M{args.num_users}-Mc{args.concurrency}-b{args.buffer_size}'
    destination = state_root / 'runs/active' / setting / args.algo / f'seed={args.seed}'
    destination = Path(external_path(destination, 'output_dir'))
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination / filename)


def _current_device_memory_mb(args):
    if getattr(args, 'device', 'cpu') != 'cpu':
        if npu_is_available():
            try:
                return torch.npu.max_memory_allocated() / (1024 ** 2)
            except Exception:
                return ''
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ''


def _current_process_memory_mb():
    current_mb = 0.0
    peak_mb = 0.0
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 ** 2)
        peak_mb = peak / (1024 ** 2)
    try:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        peak_mb = max(peak_mb, rss_mb)
    except Exception:
        pass
    return current_mb, peak_mb


def _system_metrics_fieldnames():
    return [
        'round',
        'algo',
        'dataset',
        'alpha',
        'random_cost',
        'concurrency',
        'buffer_size',
        'seed',
        'run_tag',
        'simulated_wall_time',
        'runtime_sec',
        'round_runtime_sec',
        'process_memory_mb',
        'process_peak_memory_mb',
        'device_peak_memory_mb',
        'regrouped',
        'last_recluster_time_sec',
        'total_recluster_time_sec',
        'max_recluster_time_sec',
        'recluster_count',
    ]


def system_metrics_output_path(args, output_path):
    log_dir = getattr(args, 'system_metrics_log_dir', '')
    if not log_dir:
        return None
    log_dir = external_path(log_dir, 'system_metrics_log_dir')
    output_name = os.path.basename(output_path)
    if output_name.endswith('-test_acc.txt'):
        output_name = output_name[:-len('-test_acc.txt')] + '-system_metrics.csv'
    else:
        output_name = os.path.splitext(output_name)[0] + '-system_metrics.csv'
    return os.path.join(log_dir, output_name)


def build_system_metrics_path(args, output_path):
    path = system_metrics_output_path(args, output_path)
    if path is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_metrics_fieldnames())
        writer.writeheader()
    print(f"System metrics log: {path}", flush=True)
    return path


def log_system_metrics(args, state, csv_path, train_start_time, round_runtime_sec):
    if not csv_path:
        return
    process_mb, process_peak_mb = _current_process_memory_mb()
    row = {
        'round': int(state.get('iterations', 0)),
        'algo': getattr(args, 'algo', ''),
        'dataset': getattr(args, 'dataset', ''),
        'alpha': getattr(args, 'alpha', ''),
        'random_cost': getattr(args, 'random_cost', ''),
        'concurrency': getattr(args, 'concurrency', ''),
        'buffer_size': getattr(args, 'buffer_size', ''),
        'seed': getattr(args, 'seed', ''),
        'run_tag': getattr(args, 'run_tag', ''),
        'simulated_wall_time': float(state.get('global_cost', 0.0)),
        'runtime_sec': time.perf_counter() - train_start_time,
        'round_runtime_sec': float(round_runtime_sec),
        'process_memory_mb': process_mb,
        'process_peak_memory_mb': process_peak_mb,
        'device_peak_memory_mb': _current_device_memory_mb(args),
        'regrouped': int(bool(state.get('last_regrouped', False))),
        'last_recluster_time_sec': float(state.get('last_recluster_time_sec', 0.0)),
        'total_recluster_time_sec': float(state.get('total_recluster_time_sec', 0.0)),
        'max_recluster_time_sec': float(state.get('max_recluster_time_sec', 0.0)),
        'recluster_count': int(state.get('recluster_count', 0)),
    }
    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_metrics_fieldnames())
        writer.writerow(row)


def evaluate_global_model(state, dataset_test, args):
    net_glob = state['net_glob']
    net_glob.eval()

    with torch.no_grad():
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

    net_glob.train()
    if args.algo == 'AdaMasFL' and not all(math.isfinite(v) for v in (acc_test, loss_test)):
        raise ValueError('AdaMasFL evaluation accuracy/loss is non-finite')
    return acc_test, loss_test


def log_evaluation(state, dataset_test, args, output_path, validation_dataset=None, validation_path=None):
    acc_test, loss_test = evaluate_global_model(state, dataset_test, args)
    iterations = state['iterations']

    print('Round {:3d}, Test loss {:.3f}'.format(iterations, loss_test), flush=True)
    print('Round {:3d}, Test acc {:.3f}'.format(iterations, acc_test), flush=True)

    with open(output_path, 'a') as handle:
        handle.write(str(acc_test))
        handle.write('\n')

    if validation_dataset is not None and validation_path:
        acc_val, loss_val = evaluate_global_model(state, validation_dataset, args)
        print('Round {:3d}, Validation loss {:.3f}'.format(iterations, loss_val), flush=True)
        print('Round {:3d}, Validation acc {:.3f}'.format(iterations, acc_val), flush=True)
        with open(validation_path, 'a') as handle:
            handle.write(str(acc_val))
            handle.write('\n')


def should_evaluate(state, args):
    return state['iterations'] % args.interval == 0


def run_training(
    args,
    state,
    dataset_train,
    dict_users,
    num_samples,
    dataset_test,
    output_path,
    direction_skew_path=None,
    system_metrics_path=None,
    validation_dataset=None,
    validation_path=None,
):
    train_start_time = time.perf_counter()
    if system_metrics_path and not tracemalloc.is_tracing():
        tracemalloc.start()

    while state["iterations"] < args.total_rounds:
        round_start_time = time.perf_counter()
        state = run_one_round(args, state, dataset_train, dict_users, num_samples)
        round_runtime_sec = time.perf_counter() - round_start_time
        log_direction_skew_metrics(args, state, direction_skew_path)
        log_system_metrics(args, state, system_metrics_path, train_start_time, round_runtime_sec)

        if should_evaluate(state, args):
            log_evaluation(
                state,
                dataset_test,
                args,
                output_path,
                validation_dataset=validation_dataset,
                validation_path=validation_path,
            )

        gc.collect()

    return state


def reset_evaluation_outputs(output_path, validation_path=None):
    """Exclusively reserve fresh traces before any client initialization."""
    for path in (output_path, validation_path):
        if path and os.path.lexists(path):
            raise FileExistsError(f'Existing evaluation output requires a separate run_tag or output_dir: {path}')
    for path in (output_path, validation_path):
        if path:
            with open(path, 'x'):
                pass


def resolved_arguments(args):
    runtime_fields = {'client_delay_rng', 'fedscale_trace_sampler'}
    return {key: str(value) if isinstance(value, torch.device) else value
            for key, value in vars(args).items() if key not in runtime_fields}


def source_identity():
    source_root = Path(__file__).resolve().parents[1]
    commit = os.environ.get('SAW_SOURCE_COMMIT', '').strip()
    if commit:
        if commit in ('unknown', 'unrecorded'):
            raise ValueError('SAW_SOURCE_COMMIT must identify the actual source snapshot')
        code = {'commit': commit, 'code_root': str(source_root), 'origin': 'environment'}
    else:
        def git(*tokens):
            return subprocess.check_output(
                ['git', '-C', str(source_root), *tokens], text=True,
                stderr=subprocess.PIPE).rstrip('\n')
        try:
            head = git('rev-parse', 'HEAD')
            # Include every source subdirectory, not just the entrypoint.
            status = git('status', '--porcelain=v1', '--untracked-files=all', '--', 'code')
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('Cannot detect source identity; provide SAW_SOURCE_COMMIT '
                               'for a verified source snapshot') from exc
        commit = head + ('+dirty' if status else '')
        code = {'commit': commit, 'head_commit': head, 'dirty': bool(status),
                'git_status': status.splitlines(), 'code_root': str(source_root), 'origin': 'git'}
    return {'source_commit': commit,
            'source_manifest': os.environ.get('SAW_SOURCE_MANIFEST'), 'code_identity': code}


def write_owned_identity(handle, identity):
    """Update only the file exclusively opened by this invocation."""
    owned, current = os.fstat(handle.fileno()), os.stat(handle.name, follow_symlinks=False)
    if (owned.st_dev, owned.st_ino) != (current.st_dev, current.st_ino):
        raise FileExistsError(f'Paired identity was replaced during training: {handle.name}')
    content = json.dumps(identity, indent=2)
    handle.seek(0)
    handle.write(content + '\n')
    handle.truncate()
    handle.flush()


def main():
    args = args_parser()
    configure_runtime(args)

    output_path = build_output_path(args)
    validation_path = None
    if float(getattr(args, 'validation_fraction', 0.0) or 0.0) > 0:
        validation_dir = getattr(args, 'validation_output_dir', '') or os.path.dirname(output_path)
        validation_dir = external_path(validation_dir, 'validation_output_dir')
        os.makedirs(validation_dir, exist_ok=True)
        validation_path = os.path.join(
            validation_dir,
            os.path.basename(output_path).replace('-test_acc.txt', '-validation_acc.txt'),
        )
        args.validation_output_path = validation_path
    config_name = ('resolved_config.json' if args.algo == 'AdaMasFL'
                   else Path(output_path).stem + '-resolved_config.json')
    config_path = os.path.join(os.path.dirname(output_path), config_name)
    identity_path = str(Path(output_path).with_name(Path(output_path).stem + '-identity.json'))
    metrics_path = system_metrics_output_path(args, output_path)
    metadata_path = (output_path.replace('-test_acc.txt', '-validation-metadata.json')
                     if validation_path else None)
    for path in (output_path, validation_path, config_path, identity_path, metrics_path, metadata_path):
        if path and os.path.lexists(path):
            raise FileExistsError(f'Existing run output requires a separate run_tag or output_dir: {path}')

    source = source_identity()
    command = [sys.executable, *sys.argv]
    identity = {
        'producer': 'main_fed',
        **source, 'args': resolved_arguments(args), 'cache_device': resolve_cache_device(args),
        'argv': list(sys.argv), 'command': command, 'actual_command': command,
        'executable': sys.executable, 'cwd': os.getcwd(), 'pid': os.getpid(),
        'accuracy_path': output_path, 'resolved_config_path': config_path,
        'system_metrics_path': metrics_path, 'status': 'running', 'exit_code': None,
    }
    for part in Path(output_path).parts:
        if part.startswith('attempt='):
            identity['attempt'] = part.split('=', 1)[1]

    with open(identity_path, 'x+', encoding='utf-8') as identity_handle:
        write_owned_identity(identity_handle, identity)
        try:
            reset_evaluation_outputs(output_path, validation_path)
            state, dataset_train, dataset_test, dict_users, num_samples, validation_dataset = build_training_state(args)
            resolved = {
                **source, 'args': resolved_arguments(args), 'cache_device': resolve_cache_device(args),
            }
            if args.algo == 'AdaMasFL':
                resolved.update({
                    'hyperparameters': state['adamasfl_hparams'],
                    'parameter_names': list(state['adamasfl_param_names']),
                    'bn_policy': 'selected_absolute_float_mean_integer_max',
                    'client_direction': 'beta*(grad-local_c+stale_c)+(1-beta)*stale_g',
                    'cache_device': state['adamasfl_cache_device'],
                })
            with open(config_path, 'x') as handle:
                json.dump(resolved, handle, indent=2)
            identity.update(args=resolved['args'], cache_device=resolved['cache_device'])
            write_owned_identity(identity_handle, identity)
            direction_skew_path = setup_direction_skew_monitor(
                args=args,
                state=state,
                dataset_train=dataset_train,
                dict_users=dict_users,
                output_path=output_path,
            )
            system_metrics_path = build_system_metrics_path(args, output_path)

            state = run_training(
                args=args,
                state=state,
                dataset_train=dataset_train,
                dict_users=dict_users,
                num_samples=num_samples,
                dataset_test=dataset_test,
                output_path=output_path,
                direction_skew_path=direction_skew_path,
                system_metrics_path=system_metrics_path,
                validation_dataset=validation_dataset,
                validation_path=validation_path,
            )
            if metadata_path:
                with open(metadata_path, 'x') as handle:
                    json.dump({
                        'validation_fraction': float(args.validation_fraction),
                        'validation_split_seed': int(args.validation_split_seed),
                        'validation_size': int(getattr(args, 'validation_size', 0)),
                        'training_size_after_validation_split': int(getattr(args, 'training_size_after_validation_split', 0)),
                        'validation_path': args.validation_output_path,
                        'test_used_for_selection': False,
                    }, handle, indent=2)
            identity.update(status='completed', exit_code=0, completed_rounds=int(state['iterations']))
            write_owned_identity(identity_handle, identity)
        except BaseException as exc:
            if isinstance(exc, SystemExit):
                exit_code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            else:
                exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            identity.update(status='failed', exit_code=exit_code, error_type=type(exc).__name__, error=str(exc))
            try:
                write_owned_identity(identity_handle, identity)
            except Exception as recording_error:
                print(f'Failed to record trainer exit identity: {recording_error}', file=sys.stderr, flush=True)
            raise


if __name__ == '__main__':
    main()

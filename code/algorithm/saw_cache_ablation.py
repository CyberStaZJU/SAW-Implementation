"""SAW cache-ablation variants.

All variants keep SAW's cross-client cache update schedule and only change
how cached deltas are selected or weighted during server aggregation.
"""

import copy
import math
import random

import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass

from models.local_update import LocalSGD
from utils.aggregation import buffered_aggregation
from utils.state_dict_ops import (
    load_param_dict_,
    model_param_dict,
    sd_sub,
    sd_cache_zero_like,
    sd_to_cache,
    resolve_cache_device,
)

SAW_CACHE_ABLATION_ALGOS = {
    'SAW_UniformCache',
    'SAW_FreshTopK',
    'SAW_Power',
    'SAW_Exp',
}


def _eligible_indices(delays, buffer_list):
    return [
        i for i, d in enumerate(delays)
        if d is not None or i in buffer_list
    ]


def _tau(delay):
    return float(delay) + 1.0


def _normalize(raw_weights, eligible, size):
    weights = [0.0] * size
    total = float(sum(raw_weights))
    if total <= 0.0:
        if not eligible:
            return weights
        uniform = 1.0 / float(len(eligible))
        for i in eligible:
            weights[i] = uniform
        return weights
    for i, raw in zip(eligible, raw_weights):
        weights[i] = float(raw) / total
    return weights


def _select_topk_fresh(delays, buffer_list, topk):
    eligible = _eligible_indices(delays, buffer_list)
    if topk is None or int(topk) <= 0 or int(topk) >= len(eligible):
        return eligible
    return sorted(
        eligible,
        key=lambda i: (_tau(delays[i]), i),
    )[:int(topk)]


def _cache_weights(delays, buffer_list, args):
    algo = str(args.algo)
    eligible = _eligible_indices(delays, buffer_list)
    if not eligible:
        return [0.0] * len(delays)

    if algo == 'SAW_FreshTopK':
        selected = _select_topk_fresh(
            delays,
            buffer_list,
            getattr(args, 'saw_cache_topk', 5),
        )
        return _normalize([1.0] * len(selected), selected, len(delays))

    if algo == 'SAW_Exp':
        lam = float(getattr(args, 'saw_cache_exp_lambda', 0.2))
        raw = [math.exp(-lam * _tau(delays[i])) for i in eligible]
        return _normalize(raw, eligible, len(delays))

    if algo == 'SAW_Power':
        power = float(getattr(args, 'saw_cache_power', 1.0))
        raw = [_tau(delays[i]) ** (-power) for i in eligible]
        return _normalize(raw, eligible, len(delays))

    if algo == 'SAW_UniformCache':
        return _normalize([1.0] * len(eligible), eligible, len(delays))

    raise ValueError(f"Unsupported SAW cache ablation: {algo}")


def _local_train_delta(state, args, idx):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    w_start = model_param_dict(state['net_glob'], device=args.device)
    local = LocalSGD(
        args=args,
        dataset=state['dataset_train'],
        idxs=state['dict_users'][idx],
        iters=args.local_period,
        nums=state['num_samples'][idx],
    )
    w_local = local.train(net=net)
    delta = sd_sub(w_local, w_start)
    del w_start, w_local, net
    return sd_to_cache(delta, args)


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(args.num_users)]
    state['cache'] = [None for _ in range(args.num_users)]
    state['stamp'] = [-1] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['delays'] = [None] * int(args.num_users)
    state['global_cost'] = 0
    state['iterations'] = 0

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)
    return state


def run_round(state, args, random_cost):
    active_list = [i for i, c in enumerate(state['cost']) if c > 0]
    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [
            (i, c) for i, c in enumerate(state['cost'])
            if i not in buffer_list and c > 0
        ]
        idx, _ = min(filtered, key=lambda x: x[1])
        buffer_list.append(idx)

    state['global_cost'] = state['cost'][buffer_list[-1]]

    for idx in buffer_list:
        state['cost'][idx] = -1
        state['delays'][idx] = state['iterations'] - state['stamp'][idx]
        state['cache'][idx] = state['delta'][idx]

    weights = _cache_weights(state['delays'], buffer_list, args)
    aggregated_diff = buffered_aggregation(weights, state['cache'], state['w_glob'])

    with torch.no_grad():
        for key in aggregated_diff.keys():
            state['w_glob'][key] += aggregated_diff[key]
    load_param_dict_(state['net_glob'], state['w_glob'])
    state['iterations'] += 1
    state['delays'] = [d + 1 if d is not None else None for d in state['delays']]

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:args.buffer_size]

    for idx in sampled_idx:
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)
    return state

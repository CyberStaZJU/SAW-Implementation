"""Arrival-Frozen-Cache ablation for SAW.

The cache contains every valid cached update, as in SAW. Each update's age is
measured once when it first enters the cache and remains fixed thereafter.
Weights are recomputed and normalized each aggregation, but the stored ages do
not advance after insertion. This isolates whether arrival-time freshness is
sufficient without post-arrival aging.
"""
import copy
import random

import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass

from models.local_update import LocalSGD
from utils.state_dict_ops import resolve_cache_device, sd_cache_zero_like, sd_to_cache
from utils.aggregation import buffered_aggregation
from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_sub


def _arrival_frozen_weights(ages, buffer_list):
    eligible = [i for i, age in enumerate(ages) if age is not None or i in buffer_list]
    if not eligible:
        return [0.0] * len(ages)
    weights = [0.0] * len(ages)
    raw = {i: 1.0 / max(1.0, float(ages[i])) for i in eligible}
    total = sum(raw.values())
    for i, value in raw.items():
        weights[i] = value / total
    return weights


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
    state['cache'] = [None] * int(args.num_users)
    state['stamp'] = [-1] * int(args.num_users)
    state['arrival_age'] = [None] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['global_cost'] = 0
    state['iterations'] = 0

    for idx in random.sample(range(args.num_users), args.concurrency):
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)
    return state


def run_round(state, args, random_cost):
    active_list = [i for i, cost in enumerate(state['cost']) if cost > 0]
    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [(i, cost) for i, cost in enumerate(state['cost']) if i not in buffer_list and cost > 0]
        idx, _ = min(filtered, key=lambda item: item[1])
        buffer_list.append(idx)

    state['global_cost'] = state['cost'][buffer_list[-1]]
    for idx in buffer_list:
        state['cost'][idx] = -1
        # Each replacement is a new update with its own frozen arrival age.
        state['arrival_age'][idx] = state['iterations'] - state['stamp'][idx] + 1
        state['cache'][idx] = state['delta'][idx]

    weights = _arrival_frozen_weights(state['arrival_age'], buffer_list)
    aggregated_diff = buffered_aggregation(weights, state['cache'], state['w_glob'])
    with torch.no_grad():
        for key in aggregated_diff:
            state['w_glob'][key] += aggregated_diff[key]
    load_param_dict_(state['net_glob'], state['w_glob'])
    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    for idx in sampled_idx[:args.buffer_size]:
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)
    return state

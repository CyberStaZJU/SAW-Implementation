"""SAW ablation variant: EqualW.

Keeps the cross-client cache but uses uniform aggregation over all cached
deltas that are eligible (i.e., clients with a non-None delay, plus the
freshly-filled buffer entries). Identical to SAW except that the aggregation
weights are uniform (1/N) instead of staleness-inverse.
"""

import copy
import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass
import random
from models.local_update import LocalSGD
from utils.state_dict_ops import resolve_cache_device, sd_cache_zero_like, sd_to_cache
from utils.state_dict_ops import load_param_dict_, sd_sub, model_param_dict
from utils.aggregation import buffered_aggregation


def _uniform_weights(delays, buffer_list):
    """Uniform weights over all clients with a non-None delay, plus buffer
    entries whose delay was just set. Mirror SAW's rescaling eligibility,
    but with uniform magnitude."""
    eligible = [
        i for i, d in enumerate(delays)
        if d is not None or i in buffer_list
    ]
    if not eligible:
        return [0.0] * len(delays)
    n = len(eligible)
    weights = [0.0] * len(delays)
    for i in eligible:
        weights[i] = 1.0 / n
    return weights


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(args.num_users)]
    state['cache'] = [None for _ in range(args.num_users)]
    state['stamp'] = [-1] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['delays'] = [None] * int(args.num_users)
    state['global_cost'] = 0
    state['iterations'] = 0

    def local_train_delta(idx):
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

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        state['delta'][idx] = local_train_delta(idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, random_cost):
    active_list = [i for i, c in enumerate(state['cost']) if c > 0]
    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [(i, c) for i, c in enumerate(state['cost']) if i not in buffer_list and c > 0]
        idx, _ = min(filtered, key=lambda x: x[1])
        buffer_list.append(idx)

    state['global_cost'] = state['cost'][buffer_list[-1]]

    for idx in buffer_list:
        state['cost'][idx] = -1
        state['delays'][idx] = state['iterations'] - state['stamp'][idx]
        state['cache'][idx] = state['delta'][idx]

    weights = _uniform_weights(state['delays'], buffer_list)
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

    def local_train_delta(idx):
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

    for idx in sampled_idx:
        state['delta'][idx] = local_train_delta(idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)
    return state

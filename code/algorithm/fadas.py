import copy
import random

import numpy as np
import torch

from models.local_update import LocalSGD
from utils.state_dict_ops import resolve_cache_device, sd_cache_zero_like, sd_to_cache
from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_average, sd_sub, sd_zero_like


def _trainable_param_keys(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _getattr(args, name, default):
    if not hasattr(args, name):
        return default
    value = getattr(args, name)
    return default if value is None else value


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


def _global_lr(args):
    return float(args.global_lr if args.global_lr is not None else args.lr)


def _use_delay_adaptation(args):
    return bool(_getattr(args, 'fadas_delay_adaptive', False))


def _delay_adaptive_lr(args, max_delay):
    eta = _global_lr(args)
    if not _use_delay_adaptation(args):
        return eta

    tau_c = float(_getattr(args, 'fadas_tau_c', max(1, int(args.buffer_size))))
    max_delay = max(0.0, float(max_delay))
    if max_delay <= tau_c or max_delay <= 0.0:
        return eta
    return min(eta, eta / max_delay)


def _update_adaptive_moments(state, args, aggregated_delta):
    beta1 = float(_getattr(args, 'fadas_beta1', 0.9))
    beta2 = float(_getattr(args, 'fadas_beta2', 0.99))
    beta1 = min(max(beta1, 0.0), 0.999)
    beta2 = min(max(beta2, 0.0), 0.999999)
    param_keys = state['fadas_param_keys']

    with torch.no_grad():
        for key in param_keys:
            state['fadas_m'][key].mul_(beta1).add_(aggregated_delta[key], alpha=1.0 - beta1)
            state['fadas_v'][key].mul_(beta2).addcmul_(
                aggregated_delta[key],
                aggregated_delta[key],
                value=1.0 - beta2,
            )
            state['fadas_vhat'][key] = torch.maximum(state['fadas_vhat'][key], state['fadas_v'][key])


def _apply_adaptive_update(state, args, lr, aggregated_delta):
    eps = float(_getattr(args, 'fadas_eps', 1e-8))
    param_keys = state['fadas_param_keys']

    with torch.no_grad():
        for key in state['w_glob'].keys():
            if key in param_keys:
                denom = state['fadas_vhat'][key].sqrt().add(eps)
                state['w_glob'][key].add_(state['fadas_m'][key] / denom, alpha=lr)
            else:
                state['w_glob'][key].add_(aggregated_delta[key])

    load_param_dict_(state['net_glob'], state['w_glob'])


def _sync_delays(state):
    for idx, stamp in enumerate(state['stamp']):
        if stamp < 0:
            state['delays'][idx] = 0
        else:
            state['delays'][idx] = max(0, int(state['iterations'] - stamp))


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    num_users = int(args.num_users)
    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(num_users)]
    state['stamp'] = [-1] * num_users
    state['cost'] = [-1] * num_users
    state['delays'] = [0] * num_users
    state['global_cost'] = 0
    state['iterations'] = 0

    state['fadas_m'] = sd_zero_like(state['w_glob'])
    state['fadas_v'] = sd_zero_like(state['w_glob'])
    state['fadas_vhat'] = sd_zero_like(state['w_glob'])
    state['fadas_param_keys'] = _trainable_param_keys(state['net_glob'])
    state['fadas_last_max_delay'] = 0
    state['fadas_last_lr'] = _global_lr(args)

    sampled_idx = random.sample(range(num_users), min(int(args.concurrency), num_users))
    for idx in sampled_idx:
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, random_cost):
    active_list = [i for i, cost in enumerate(state['cost']) if cost > 0]
    buffer_list = []

    for _ in range(args.buffer_size):
        filtered = [
            (idx, cost)
            for idx, cost in enumerate(state['cost'])
            if idx not in buffer_list and cost > 0
        ]
        if not filtered:
            break
        idx, _ = min(filtered, key=lambda item: item[1])
        buffer_list.append(idx)

    if not buffer_list:
        return state

    state['global_cost'] = state['cost'][buffer_list[-1]]
    max_delay = 0
    for idx in buffer_list:
        state['cost'][idx] = -1
        delay = max(0, int(state['iterations'] - state['stamp'][idx]))
        state['delays'][idx] = delay
        max_delay = max(max_delay, delay)
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    aggregated_delta = sd_average([state['delta'][idx] for idx in buffer_list], ref_model=state['w_glob'])
    _update_adaptive_moments(state, args, aggregated_delta)

    step_lr = _delay_adaptive_lr(args, max_delay)
    _apply_adaptive_update(state, args, step_lr, aggregated_delta)
    state['fadas_last_max_delay'] = max_delay
    state['fadas_last_lr'] = step_lr

    state['iterations'] += 1
    _sync_delays(state)

    sampled_idx = [idx for idx in range(args.num_users) if idx not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(buffer_list)]

    for idx in sampled_idx:
        state['delta'][idx] = _local_train_delta(state, args, idx)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)
        state['delays'][idx] = 0

    return state

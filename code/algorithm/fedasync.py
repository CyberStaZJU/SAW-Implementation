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
from utils.aggregation import buffered_aggregation
from utils.state_dict_ops import load_param_dict_


def _getattr(args, name, default):
    if not hasattr(args, name):
        return default
    value = getattr(args, name)
    return default if value is None else value


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _fedasync_alpha(args, staleness):
    base_alpha = _clamp01(_getattr(args, 'fedasync_alpha', 0.2))
    mode = str(_getattr(args, 'fedasync_staleness_func', 'constant')).lower()
    tau = max(0.0, float(staleness))

    if mode == 'poly':
        exponent = max(0.0, float(_getattr(args, 'fedasync_poly_a', 0.5)))
        scale = (tau + 1.0) ** (-exponent)
    elif mode == 'hinge':
        a = max(0.0, float(_getattr(args, 'fedasync_hinge_a', 10.0)))
        b = max(0.0, float(_getattr(args, 'fedasync_hinge_b', 4.0)))
        scale = 1.0 if tau <= b else 1.0 / (a * (tau - b) + 1.0)
    else:
        scale = 1.0

    return _clamp01(base_alpha * scale)


def _local_train_model(state, args, idx, dataset_train, dict_users, num_samples):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    local = LocalSGD(
        args=args,
        dataset=dataset_train,
        idxs=dict_users[idx],
        iters=args.local_period,
        nums=num_samples[idx],
        rho=float(_getattr(args, 'fedasync_rho', 0.1)),
    )
    w_local = local.train(net=net)
    del net
    return sd_to_cache(w_local, args)


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delays'] = [0 for _ in range(args.num_users)]
    state['stamp'] = [0 for _ in range(args.num_users)]
    state['cost'] = [-1 for _ in range(args.num_users)]
    state['local_models'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(args.num_users)]
    state['iterations'] = 0

    sampled_idx = random.sample(range(args.num_users), min(int(args.concurrency), int(args.num_users)))
    for idx in sampled_idx:
        state['local_models'][idx] = _local_train_model(
            state,
            args,
            idx,
            state['dataset_train'],
            state['dict_users'],
            state['num_samples'],
        )
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, dataset_train, dict_users, num_samples, random_cost):
    active_list = [i for i, c in enumerate(state['cost']) if c > 0]
    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [(i, c) for i, c in enumerate(state['cost']) if i not in buffer_list and c > 0]
        if not filtered:
            break
        idx, _ = min(filtered, key=lambda x: x[1])
        buffer_list.append(idx)

    if len(buffer_list) == 0:
        state['last_buffer_list'] = []
        state['last_valid_buffer_list'] = []
        state['last_invalid_buffer_list'] = []
        state['last_selected_item_summary'] = []
        state['last_regrouped'] = False
        return state

    state['global_cost'] = state['cost'][buffer_list[-1]]
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    for idx in buffer_list:
        state['cost'][idx] = -1
        state['delays'][idx] = state['iterations'] - state['stamp'][idx]

    alphas = [_fedasync_alpha(args, state['delays'][idx]) for idx in buffer_list]
    alpha_sum = sum(alphas)
    if alpha_sum <= 0:
        weights = [1.0 / len(buffer_list) for _ in buffer_list]
        alpha = 0.0
    else:
        weights = [alpha / alpha_sum for alpha in alphas]
        alpha = alpha_sum / float(len(buffer_list))

    selected_models = [state['local_models'][i] for i in buffer_list]
    aggregated_model = buffered_aggregation(weights, selected_models, state['w_glob'])

    with torch.no_grad():
        for key in aggregated_model.keys():
            state['w_glob'][key] = (1.0 - alpha) * state['w_glob'][key] + alpha * aggregated_model[key]

    load_param_dict_(state['net_glob'], state['w_glob'])
    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(buffer_list)]

    for idx in sampled_idx:
        state['local_models'][idx] = _local_train_model(
            state,
            args,
            idx,
            dataset_train,
            dict_users,
            num_samples,
        )
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)

    return state

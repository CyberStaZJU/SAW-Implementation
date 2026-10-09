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
from utils.state_dict_ops import sd_sub, model_param_dict, load_param_dict_, sd_average


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(args.num_users)]
    state['stamp'] = [0] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['iterations'] = 0

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        net = copy.deepcopy(state['net_glob']).to(args.device)
        w_start = model_param_dict(state['net_glob'])
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

        state['delta'][idx] = sd_to_cache(delta, args)
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
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    selected_updates = [state['delta'][i] for i in buffer_list]
    aggregated_diff = sd_average(selected_updates, ref_model=state['w_glob'])

    with torch.no_grad():
        for key in aggregated_diff.keys():
            state['w_glob'][key] += aggregated_diff[key]
    load_param_dict_(state['net_glob'], state['w_glob'])
    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:args.buffer_size]

    for idx in sampled_idx:
        net = copy.deepcopy(state['net_glob']).to(args.device)
        w_start = model_param_dict(state['net_glob'])
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

        state['delta'][idx] = sd_to_cache(delta, args)
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)

    return state

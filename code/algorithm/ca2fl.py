# algorithms/ca2fl_algo.py
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
from utils.state_dict_ops import sd_sub, load_param_dict_, model_param_dict, sd_zero_like


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delta'] = [sd_cache_zero_like(state["w_glob"], args) for _ in range(args.num_users)]
    state['cache'] = [sd_cache_zero_like(state["w_glob"], args) for _ in range(args.num_users)]
    state['cost'] = [-1 for _ in range(args.num_users)]
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
        state['cost'][idx] = random_cost(idx)
    
    return state


def run_round(state, args, dataset_train, dict_users, num_samples, random_cost):
    cost = state["cost"]
    buffer_list = []
    eta = 1
    server_v = sd_zero_like(state["w_glob"])

    active_list = [i for i, c in enumerate(cost) if c > 0]

    for _ in range(args.buffer_size):
        filtered_costs = [(i, c) for i, c in enumerate(cost) if i not in buffer_list and c > 0]
        idx, _ = min(filtered_costs, key=lambda x: x[1])
        buffer_list.append(idx)

    state["global_cost"] = cost[buffer_list[-1]]

    for idx in buffer_list:
        cost[idx] = -1
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    for i in range(args.num_users):
        if state['cache'][i] is not None:
            for key in server_v.keys():
                server_v[key] += state['cache'][i][key].to(server_v[key].device) / args.num_users
            if i in buffer_list:
                for key in server_v.keys():
                    server_v[key] -= state['cache'][i][key].to(server_v[key].device) / args.buffer_size

    weights = [1.0 / int(args.buffer_size) for _ in range(args.buffer_size)]
    selected_updates = [state["delta"][i] for i in buffer_list]
    aggregated_diff = buffered_aggregation(weights, selected_updates, state["w_glob"])

    with torch.no_grad():
        for key in aggregated_diff.keys():
            server_v[key] += aggregated_diff[key]
            state["w_glob"][key] += eta * server_v[key]

    load_param_dict_(state["net_glob"], state["w_glob"])

    for idx in buffer_list:
        state['cache'][idx] = state["delta"][idx]

    state["iterations"] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:args.buffer_size]

    for idx in sampled_idx:
        net = copy.deepcopy(state["net_glob"]).to(args.device)
        w_start = model_param_dict(state['net_glob'], device=args.device)

        local = LocalSGD(
            args=args,
            dataset=dataset_train,
            idxs=dict_users[idx],
            iters=args.local_period,
            nums=num_samples[idx]
        )
        w = local.train(net=net)

        state["delta"][idx] = sd_to_cache(sd_sub(w, w_start), args)
        cost[idx] = state["global_cost"] + random_cost(idx)

        del w_start, w, net

    state["cost"] = cost

    return state

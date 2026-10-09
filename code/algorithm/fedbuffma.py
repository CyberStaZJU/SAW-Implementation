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
from utils.state_dict_ops import load_param_dict_, sd_sub, sd_axpy, sd_copy, sd_zero_like, model_param_dict

class MomentumApproximation(object):
    """
    Online momentum approximation for buffered asynchronous FL.

    MA-Light combines the current update with the previous server momentum.
    """
    def __init__(self, args, beta=0.9, p=0.0):
        self.T = args.total_rounds
        self.beta = float(beta)
        self.p = float(p)
        self.args = args

        # Round statistics for the light least-squares recurrence.
        self.W = np.zeros((self.T, self.T), dtype=np.float64)
        self.light_A = np.zeros((self.T+1, self.T+1), dtype=np.float64)

        # Previous light-weight momentum
        self.prev_a = None
        self.prev_m = None
        self.last_u = 1.0
        self.last_v = 0.0

    def build_round_statistics(self, current_round, selected_clients, stamps, scales, normalizer):

        row = np.zeros(self.T, dtype=np.float64)
        denom = max(float(normalizer), 1.0)
        for idx, scale in zip(selected_clients, scales):
            col = max(0, stamps[idx])
            row[col] += float(scale / denom)

        self.W[current_round, :] = row

    def solve_light(self, iterations, r_t):
        W_sub = self.W[:iterations+1, :iterations+1]
        # target = self.M_row(self.args, iterations)
        target = np.zeros(iterations + 1, dtype=np.float64)
        for s_idx in range(iterations + 1):
                target[s_idx] = (1.0 - self.beta) * (self.beta ** (iterations - s_idx))

        if self.prev_a is None:
            u = 1.0
            v = 0.0
            a_t = np.zeros(iterations+1, dtype=np.float64)
            a_t[-1] = u
            m_t = sd_copy(r_t)
        else:
            prev_ext = np.zeros(iterations+1, dtype=np.float64)
            prev_length = min(len(self.prev_a), iterations)
            prev_ext[:prev_length] = self.prev_a[:prev_length]

            basis_curr = W_sub[iterations, :]
            basis_prev = prev_ext @ W_sub
            B = np.stack([basis_curr, basis_prev], axis=1)

            coeffs = np.linalg.lstsq(B, target, rcond=None)[0]
            u = float(coeffs[0])
            v = float(coeffs[1])

            a_t = u * np.eye(iterations+1, dtype=np.float64)[iterations] + v * prev_ext
            m_t = sd_copy(self.prev_m)
            for k in m_t.keys():
                m_t[k] = u * r_t[k] + v * m_t[k]

        self.light_A[iterations+1, :iterations+1] = a_t
        self.prev_a = a_t.copy()
        self.prev_m = sd_copy(m_t)
        self.last_u = float(u)
        self.last_v = float(v)
        return m_t, {'a_tilde': a_t, 'u': u, 'v': v, 'target': target, 'W_sub': W_sub}

def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(args.num_users)]
    state['stamp'] = [0] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['iterations'] = 0
    ma_beta = float(getattr(args, 'momentum', 0.9))
    ma_p = float(getattr(args, 'ma_p', getattr(args, 'staleness_power', 0.0)))
    state['ma_solver'] = MomentumApproximation(
        args,
        beta=ma_beta,
        p=ma_p
    )

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
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
        state['cost'][idx] = random_cost(idx)

    return state

def run_round(state, args, random_cost):
    active_list = [i for i, c in enumerate(state['cost']) if c > 0]
    buffer_list = []
    consumed_list = []
    scale_list = []
    last_cost = state['global_cost']
    max_staleness = getattr(args, 'ma_staleness_limit', None)
    if max_staleness is not None and max_staleness < 0:
        max_staleness = None
    ma_p = float(getattr(args, 'ma_p', 0.0))

    while len(buffer_list) < args.buffer_size:
        filtered = [(i, c) for i, c in enumerate(state['cost']) if i not in consumed_list and c > 0]
        if not filtered:
            break
        idx, client_cost = min(filtered, key=lambda x: x[1])
        consumed_list.append(idx)
        last_cost = client_cost

        tau = max(0, state['iterations'] - state['stamp'][idx])
        state['cost'][idx] = -1
        if max_staleness is not None and tau > max_staleness:
            continue

        buffer_list.append(idx)
        scale_list.append((tau + 1.0) ** (-ma_p))

    state['global_cost'] = last_cost
    state['last_buffer_list'] = list(consumed_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = [idx for idx in consumed_list if idx not in buffer_list]
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    if not buffer_list:
        sampled_idx = [i for i in range(args.num_users) if i not in active_list]
        np.random.shuffle(sampled_idx)
        sampled_idx = sampled_idx[:len(consumed_list)]
        for idx in sampled_idx:
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

            state['delta'][idx] = sd_to_cache(delta, args)
            state['stamp'][idx] = state['iterations']
            state['cost'][idx] = state['global_cost'] + random_cost(idx)
        return state

    r_t = sd_zero_like(state['w_glob'])
    normalizer = float(args.buffer_size)
    for idx, scale in zip(buffer_list, scale_list):
        sd_axpy(r_t, scale / normalizer, state['delta'][idx])

    ma_solver = state['ma_solver']
    ma_solver.build_round_statistics(
        state['iterations'],
        selected_clients=buffer_list,
        stamps=state['stamp'],
        scales=scale_list,
        normalizer=normalizer,
    )

    simulated_momentum, _ = ma_solver.solve_light(state['iterations'], r_t)
    with torch.no_grad():
        for key in simulated_momentum.keys():
            state['w_glob'][key] += simulated_momentum[key]

    load_param_dict_(state['net_glob'], state['w_glob'])

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(consumed_list)]
    state['iterations'] += 1
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

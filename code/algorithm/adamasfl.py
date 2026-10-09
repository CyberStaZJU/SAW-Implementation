"""AdaMasFL, Algorithms 3-4 in Momentum-Driven Adaptivity."""
import copy
import math
import random

import numpy as np
import torch

from models.local_update import LocalSGD, LocalAdaMasFL
from utils.state_dict_ops import sd_zero_like, sd_sub, sd_axpy, sd_copy, load_param_dict_, sd_average, resolve_cache_device


def _param_state(net):
    return {name: param.detach().clone() for name, param in net.named_parameters()
            if param.requires_grad}


def _buffer_state(net):
    full_state = net.state_dict()
    return {name: full_state[name].detach().clone() for name, _ in net.named_buffers()
            if name in full_state}


def _cache_state(state, values):
    if state.get('adamasfl_cache_device', 'device') == 'cpu':
        return {name: value.detach().to('cpu', copy=True) for name, value in values.items()}
    return values


def _compute_state(state, values):
    device = next(state['net_glob'].parameters()).device
    return {name: value.to(device) for name, value in values.items()}


@torch.no_grad()
def _initial_control_mean(state, params):
    if state['adamasfl_cache_device'] == 'device':
        return sd_average(state['adamasfl_c_local'])
    # Preserve sd_average's add-then-scale order on the compute device.
    averaged = sd_zero_like(params)
    for control in state['adamasfl_c_local']:
        for name in averaged:
            averaged[name].add_(control[name].to(averaged[name].device))
    for value in averaged.values():
        value.mul_(1.0 / len(state['adamasfl_c_local']))
    return averaged


def _validate_state(net, values, label):
    # Combine checks on-device so each validation needs one host synchronization.
    if values and not torch.stack([torch.isfinite(v).all() for v in values.values()]).all().item():
        for name, value in values.items():
            if not torch.isfinite(value).all().item():
                raise ValueError(f'AdaMasFL {label}: non-finite tensor {name}')
    for prefix, module in net.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        prefix = prefix + '.' if prefix else ''
        for suffix in ('running_var', 'num_batches_tracked'):
            name = prefix + suffix
            if name in values and (values[name] < 0).any().item():
                raise ValueError(f'AdaMasFL {label}: negative {name}')


def _validate_matching_state(net, reference, values, label):
    if values.keys() != reference.keys():
        raise ValueError(f'AdaMasFL {label}: state keys do not match')
    for name, value in values.items():
        if value.shape != reference[name].shape or value.dtype != reference[name].dtype:
            raise ValueError(f'AdaMasFL {label}: shape/dtype mismatch for {name}')
    _validate_state(net, values, label)


def _expand_to_param_state(ref_state, partial_state):
    param_state = sd_zero_like(ref_state)
    if partial_state is not None:
        for key, value in partial_state.items():
            if key in param_state:
                param_state[key] = value.detach().clone()
    return param_state


class _FiniteLocalLoss:
    def _compute_loss_and_backward(self, net, outputs, init_params=None):
        if not torch.isfinite(outputs.loss).all().item():
            raise ValueError('AdaMasFL local loss is non-finite')
        loss = super()._compute_loss_and_backward(net, outputs, init_params)
        grads = {name: param.grad for name, param in self._named_trainable_params(net)
                 if param.grad is not None}
        _validate_state(net, grads, 'local gradients')
        return loss


class _CheckedLocalSGD(_FiniteLocalLoss, LocalSGD):
    pass


class _CheckedLocalAdaMasFL(_FiniteLocalLoss, LocalAdaMasFL):
    def modify_gradients(self, net):
        grads = {name: param.grad for name, param in self._named_trainable_params(net)
                 if param.grad is not None}
        _validate_state(net, grads, 'local gradients')
        super().modify_gradients(net)
        directions = {name: self.g_next[name] for name in grads}
        _validate_state(net, directions, 'local direction')
        sqsum = sum(torch.sum(value ** 2) for value in directions.values())
        if not torch.isfinite(sqsum).item():
            raise ValueError('AdaMasFL local direction norm is non-finite')
        _validate_state(net, {name: param.grad for name, param in self._named_trainable_params(net)
                              if param.grad is not None}, 'normalized local gradients')


def _estimate_initial_control(state, args, idx):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    local = _CheckedLocalSGD(
        args=args,
        dataset=state['dataset_train'],
        idxs=state['dict_users'][idx],
        iters=args.local_period,
        nums=state['num_samples'][idx],
    )
    avg_grad = local.average_stochastic_gradients(net=net)
    _validate_state(net, avg_grad, f'initial control client {idx}')
    _validate_state(net, _buffer_state(net), f'initial estimation buffers client {idx}')
    return _expand_to_param_state(_param_state(state['net_glob']), avg_grad)


def _resolve_adamasfl_hparams(args):
    S = float(args.buffer_size)
    K = float(args.local_period)
    T = float(args.total_rounds)

    beta = (S * K / T) ** 0.5
    gamma = (S * K) ** 0.25 / (T ** 0.75)
    lr = 1.0 / (K * (T ** 0.5))

    beta_override = getattr(args, "adamasfl_beta", None)
    gamma_override = getattr(args, "adamasfl_gamma", None)
    lr_override = getattr(args, "adamasfl_lr", None)

    if beta_override is None:
        beta *= float(getattr(args, "adamasfl_beta_scale", 1.0))
    else:
        beta = float(beta_override)

    if gamma_override is None:
        gamma *= float(getattr(args, "adamasfl_gamma_scale", 1.0))
    else:
        gamma = float(gamma_override)

    if lr_override is None:
        lr *= float(getattr(args, "adamasfl_lr_scale", 1.0))
    else:
        lr = float(lr_override)

    if not all(math.isfinite(v) for v in (beta, gamma, lr)) or not 0 < beta < 1 or gamma <= 0 or lr <= 0:
        raise ValueError('AdaMasFL requires finite 0 < beta < 1, eta > 0, gamma > 0')

    return beta, gamma, lr


def _launch_adamasfl_job(state, args, idx, base_cost, random_cost):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    # Only this private local copy changes the shared export helper's mode.
    net._asyncbuffer_trainable_state_only = True
    w_start = _param_state(net)
    _validate_state(net, w_start, f'launch parameters client {idx}')
    _validate_state(net, _buffer_state(net), f'launch buffers client {idx}')

    # Keep stale global control and momentum separate for Algorithm 4.
    server_c = sd_copy(state['adamasfl_c_global'])
    g_mix = sd_copy(state['adamasfl_g'])
    local_c = _compute_state(state, state['adamasfl_c_local'][idx])
    for label, values in (('local control', local_c),
                          ('global control', server_c), ('momentum', g_mix)):
        _validate_matching_state(net, w_start, values, f'{label} client {idx}')

    local = _CheckedLocalAdaMasFL(
        args=args,
        dataset=state['dataset_train'],
        idxs=state['dict_users'][idx],
        iters=args.local_period,
        nums=state['num_samples'][idx],
        momentum_beta=state['beta'],
        local_c=local_c,
        g_mix=g_mix,
        server_c=server_c,
    )
    w_local = local.train(net=net)
    buffers = _buffer_state(net)
    _validate_matching_state(net, w_start, w_local, f'local parameters client {idx}')
    _validate_state(net, buffers, f'local buffers client {idx}')
    ec = _expand_to_param_state(w_start, local.last_ec)
    _validate_matching_state(net, w_start, ec, f'local control estimate client {idx}')
    delta = sd_sub(w_local, w_start)
    _validate_state(net, delta, f'local delta client {idx}')

    cached_ec, cached_buffers, cached_delta = (
        _cache_state(state, ec), _cache_state(state, buffers), _cache_state(state, delta))
    state['adamasfl_ec_local'][idx] = cached_ec
    state['adamasfl_buffer_local'][idx] = cached_buffers
    state['delta'][idx] = cached_delta
    state['cost'][idx] = base_cost + random_cost(idx)


def init_state(state, args, random_cost):
    net = state['net_glob']
    full_state = sd_copy(net.state_dict())
    full_state.update(sd_copy(state['w_glob']))
    _validate_state(net, full_state, 'initial model')
    load_param_dict_(net, full_state)
    state['w_glob'] = full_state
    params = _param_state(net)
    state['adamasfl_param_names'] = tuple(params)
    state['cache_device'] = resolve_cache_device(args)
    state['adamasfl_cache_device'] = state['cache_device']
    if state['adamasfl_cache_device'] not in ('device', 'cpu'):
        raise ValueError('AdaMasFL cache device must be device or cpu')
    initial_control = None
    initial_hparams = None
    if state['adamasfl_cache_device'] == 'cpu' and args.num_users > 0:
        # Keep population-sized CPU caches out of the first backend training call.
        initial_hparams = _resolve_adamasfl_hparams(args)
        args.lr = initial_hparams[2]
        initial_control = _estimate_initial_control(state, args, 0)
    cached_params = _cache_state(state, params)
    state['delta'] = [sd_zero_like(cached_params) for _ in range(args.num_users)]
    state['adamasfl_buffer_local'] = [{} for _ in range(args.num_users)]
    state['cost'] = [-1 for _ in range(args.num_users)]
    state['iterations'] = 0

    state['beta'], state['gamma'], args.lr = (
        initial_hparams if initial_hparams is not None else _resolve_adamasfl_hparams(args))
    state['adamasfl_hparams'] = {'beta': state['beta'], 'gamma': state['gamma'], 'eta': args.lr}

    state['adamasfl_c_local'] = [sd_zero_like(cached_params) for _ in range(args.num_users)]
    state['adamasfl_ec_local'] = [sd_zero_like(cached_params) for _ in range(args.num_users)]
    state['adamasfl_c_global'] = sd_zero_like(params)
    state['adamasfl_g'] = sd_zero_like(params)

    for idx in range(args.num_users):
        control = (initial_control if idx == 0 and initial_control is not None
                   else _estimate_initial_control(state, args, idx))
        state['adamasfl_c_local'][idx] = _cache_state(state, control)
        state['adamasfl_ec_local'][idx] = sd_copy(state['adamasfl_c_local'][idx])
        initial_control = None
        del control

    state['adamasfl_c_global'] = _initial_control_mean(state, params)
    state['adamasfl_g'] = sd_copy(state['adamasfl_c_global'])
    _validate_state(net, state['adamasfl_c_global'], 'initial global control')
    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        _launch_adamasfl_job(state, args, idx, base_cost=0.0, random_cost=random_cost)

    return state


def _average_buffers(net, selected_buffers):
    reference = _buffer_state(net)
    for buffers in selected_buffers:
        _validate_matching_state(net, reference, buffers, 'selected buffers')
    averaged = {}
    for name, value in reference.items():
        if value.is_floating_point() or value.is_complex():
            result = torch.zeros_like(value)
            for buffers in selected_buffers:
                result.add_(buffers[name], alpha=1.0 / len(selected_buffers))
        else:
            # Integer counters and discrete buffers use absolute elementwise maxima.
            result = selected_buffers[0][name].detach().clone()
            for buffers in selected_buffers[1:]:
                result = torch.maximum(result, buffers[name])
        averaged[name] = result
    _validate_state(net, averaged, 'averaged buffers')
    return averaged


def run_round(state, args, dataset_train, dict_users, num_samples, random_cost):
    del dataset_train, dict_users, num_samples

    cost = state['cost']
    global_cost = state['global_cost']
    beta = state['beta']
    gamma = state['gamma']
    net = state['net_glob']
    params = {name: state['w_glob'][name] for name in state['adamasfl_param_names']}

    active_list = [i for i, c in enumerate(cost) if c > 0]
    buffer_list = []

    for _ in range(args.buffer_size):
        filtered_costs = [(i, c) for i, c in enumerate(cost) if i not in buffer_list and c > 0]
        if not filtered_costs:
            break
        idx, _ = min(filtered_costs, key=lambda x: x[1])
        buffer_list.append(idx)

    if not buffer_list:
        state['last_buffer_list'] = []
        state['last_valid_buffer_list'] = []
        state['last_invalid_buffer_list'] = []
        state['last_selected_item_summary'] = []
        state['last_regrouped'] = False
        return state

    global_cost = cost[buffer_list[-1]]
    _validate_state(net, state['w_glob'], 'global model')
    _validate_matching_state(net, params, state['adamasfl_c_global'], 'global control')
    _validate_matching_state(net, params, state['adamasfl_g'], 'global momentum')
    buffers = _average_buffers(net, [_compute_state(state, state['adamasfl_buffer_local'][idx])
                                     for idx in buffer_list])
    delta_c_avg = sd_zero_like(params)
    delta_c_sum_for_c = sd_zero_like(params)
    new_controls = {}

    for idx in buffer_list:
        c_old = _compute_state(state, state['adamasfl_c_local'][idx])
        c_new = _compute_state(state, state['adamasfl_ec_local'][idx])
        for label, values in (('old control', c_old), ('new control', c_new),
                              ('delta', _compute_state(state, state['delta'][idx]))):
            _validate_matching_state(net, params, values, f'selected {label} client {idx}')
        new_controls[idx] = _cache_state(state, sd_copy(c_new))
        diff = sd_sub(c_new, c_old)
        sd_axpy(delta_c_avg, 1.0, diff)
        sd_axpy(delta_c_sum_for_c, 1.0, diff)

    S_eff = float(len(buffer_list))
    for k in delta_c_avg:
        delta_c_avg[k] /= S_eff
        delta_c_sum_for_c[k] /= float(args.num_users)

    tmp = sd_copy(state['adamasfl_c_global'])
    sd_axpy(tmp, 1.0, delta_c_avg)
    g_next = {k: beta * tmp[k] + (1.0 - beta) * state['adamasfl_g'][k] for k in params}
    c_next = sd_copy(state['adamasfl_c_global'])
    sd_axpy(c_next, 1.0, delta_c_sum_for_c)
    _validate_state(net, g_next, 'updated global momentum')
    _validate_state(net, c_next, 'updated global control')

    g_dir = sd_zero_like(params)
    for idx in buffer_list:
        delta = _compute_state(state, state['delta'][idx])
        for k in g_dir:
            g_dir[k] += (-delta[k]) / (args.lr * args.local_period * S_eff)
    _validate_state(net, g_dir, 'server direction')

    with torch.no_grad():
        updated = sd_copy(state['w_glob'])
        for k in params:
            updated[k] = params[k] - gamma * g_dir[k]
        # Running statistics are absolute convex averages, independent of eta/K/gamma.
        updated.update(buffers)
    _validate_state(net, updated, 'updated model')
    load_param_dict_(net, updated)
    state['w_glob'] = updated
    state['adamasfl_g'] = g_next
    state['adamasfl_c_global'] = c_next
    for idx, c_new in new_controls.items():
        state['adamasfl_c_local'][idx] = c_new
        cost[idx] = -1

    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False
    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(buffer_list)]

    for idx in sampled_idx:
        _launch_adamasfl_job(state, args, idx, base_cost=global_cost, random_cost=random_cost)

    state['cost'] = cost
    state['global_cost'] = global_cost
    return state

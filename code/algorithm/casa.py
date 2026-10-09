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
from utils.state_dict_ops import resolve_cache_device, sd_cache_zero_like, sd_to_cache
from utils.state_dict_ops import (
    load_param_dict_,
    model_param_dict,
    sd_axpy,
    sd_sub,
    sd_zero_like,
)


# CASA-inspired FedBuff-compatible variant:
# 1. keep the repository's buffered asynchronous outer loop,
# 2. maintain a buffer-aided client similarity matrix for dynamic clustering,
# 3. use CASA-style bi-level decay during buffered aggregation,
# 4. keep a single global model for compatibility with the current evaluation flow.


def _getattr(args, name, default):
    if not hasattr(args, name):
        return default
    value = getattr(args, name)
    return default if value is None else value


def _select_buffer(costs, buffer_size):
    buffer_list = []
    for _ in range(buffer_size):
        filtered = [
            (idx, cost)
            for idx, cost in enumerate(costs)
            if idx not in buffer_list and cost > 0
        ]
        if not filtered:
            break
        idx, _ = min(filtered, key=lambda item: item[1])
        buffer_list.append(idx)
    return buffer_list


def _float_tensor_items(sd):
    return [
        (key, value)
        for key, value in sd.items()
        if torch.is_tensor(value) and torch.is_floating_point(value)
    ]


def _build_sketch_plan(ref_sd, feature_dim):
    items = [
        (key, int(value.numel()))
        for key, value in _float_tensor_items(ref_sd)
        if int(value.numel()) > 0
    ]
    if not items:
        return []

    feature_dim = max(1, int(feature_dim))
    items.sort(key=lambda item: item[1], reverse=True)
    selected = items[:min(len(items), feature_dim)]

    alloc = [1] * len(selected)
    remaining = feature_dim - len(selected)
    if remaining > 0:
        weights = np.array([size for _, size in selected], dtype=np.float64)
        total = max(weights.sum(), 1.0)
        expected = remaining * weights / total
        extra = np.floor(expected).astype(np.int64)
        alloc = [base + int(add) for base, add in zip(alloc, extra)]

        leftover = int(remaining - extra.sum())
        if leftover > 0:
            order = np.argsort(-(expected - extra))
            for pos in order[:leftover]:
                alloc[int(pos)] += 1

    plan = []
    for (key, size), count in zip(selected, alloc):
        count = min(int(count), size)
        if count <= 0:
            continue
        if count == 1:
            idx = torch.tensor([size // 2], dtype=torch.long)
        else:
            idx = torch.linspace(0, size - 1, steps=count).round().long().unique(sorted=True)
        plan.append((key, idx))
    return plan


def _encode_delta(delta, sketch_plan):
    if not sketch_plan:
        return torch.ones(1, dtype=torch.float32)

    pieces = []
    for key, idx in sketch_plan:
        flat = delta[key].detach().float().cpu().view(-1)
        pieces.append(flat.index_select(0, idx))

    feature = torch.cat(pieces, dim=0)
    norm = torch.norm(feature, p=2)
    if norm > 0:
        feature = feature / norm
    return feature


def _feature_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(torch.clamp(torch.dot(a, b), min=-1.0, max=1.0).item())


def _compute_similarity_matrix(features):
    if not features:
        return torch.eye(1, dtype=torch.float32)

    matrix = torch.stack(features, dim=0)
    norms = torch.norm(matrix, dim=1, keepdim=True)
    matrix = matrix / torch.clamp(norms, min=1e-12)
    sim = matrix @ matrix.t()
    sim.fill_diagonal_(1.0)
    return sim.cpu()


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
    return delta


def _omega_t(round_idx, args):
    return math.exp(-float(_getattr(args, 'casa_omega', 0.01)) * float(round_idx))


def _cluster_decay(round_idx, cluster_size, args):
    alpha0 = float(_getattr(args, 'casa_alpha0', 1.0))
    cluster_bias = float(_getattr(args, 'casa_cluster_bias', 3.0))
    denom = max(math.log(max(2.0, float(cluster_size) + cluster_bias)), 1.0)
    alpha = alpha0 * _omega_t(round_idx, args) / denom
    return float(max(0.0, min(1.0, alpha)))


def _client_decay(round_idx, cluster_size, staleness, args):
    alpha_c = _cluster_decay(round_idx, cluster_size, args)
    omega_t = _omega_t(round_idx, args)
    threshold = max(1.0, float(cluster_size) * (2.0 - omega_t))
    if staleness <= threshold:
        alpha_i = alpha_c
    else:
        alpha_i = alpha_c / math.sqrt(max(1.0, float(staleness)))
    return float(max(0.0, min(1.0, alpha_i))), alpha_c, threshold


def _consume_similarity_buffer(state, idx):
    if state['client_features'][idx] is None:
        state['client_buffers'][idx] = []
        return

    for entry in state['client_buffers'][idx]:
        owner = entry['owner']
        sim = _feature_similarity(state['client_features'][idx], entry['feature'])
        state['similarity'][idx, owner] = sim
        state['similarity'][owner, idx] = sim
        state['similarity_round'][idx, owner] = state['iterations']
        state['similarity_round'][owner, idx] = state['iterations']

    state['similarity'][idx, idx] = 1.0
    state['similarity_round'][idx, idx] = state['iterations']
    state['client_buffers'][idx] = []


def _upsert_buffer_entry(buffer_entries, owner_idx, owner_stamp, feature, priority_alpha):
    record = {
        'owner': owner_idx,
        'stamp': owner_stamp,
        'feature': feature.clone(),
        'priority_alpha': float(priority_alpha),
    }

    for pos, entry in enumerate(buffer_entries):
        if entry['owner'] == owner_idx:
            if owner_stamp >= entry['stamp']:
                buffer_entries[pos] = record
            return

    buffer_entries.append(record)


def _prune_similarity_buffers(state, args):
    budget = int(_getattr(
        args,
        'casa_buffer_budget',
        max(int(args.num_users), int(args.buffer_size) * int(args.num_users)),
    ))
    if budget <= 0:
        for idx in range(len(state['client_buffers'])):
            state['client_buffers'][idx] = []
        return

    entries = []
    total = 0
    for target, buf in enumerate(state['client_buffers']):
        for pos, entry in enumerate(buf):
            last_round = int(state['similarity_round'][target, entry['owner']])
            age = state['iterations'] - last_round if last_round >= 0 else state['iterations'] + 1
            missing = 1 if last_round < 0 else 0
            entries.append((missing, age, entry['priority_alpha'], target, pos))
            total += 1

    if total <= budget:
        return

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    remove_num = total - budget
    removals = {}
    for _, _, _, target, pos in entries[:remove_num]:
        removals.setdefault(target, set()).add(pos)

    for target, positions in removals.items():
        state['client_buffers'][target] = [
            entry
            for pos, entry in enumerate(state['client_buffers'][target])
            if pos not in positions
        ]


def _insert_arrival_into_buffers(state, args, idx, priority_alpha):
    group_id = state['group_ids'][idx]
    if group_id < 0 or group_id >= len(state['group_members']):
        return

    gap_limit = _getattr(args, 'casa_start_gap', args.local_period)
    owner_stamp = state['stamp'][idx]
    feature = state['client_features'][idx]
    if feature is None:
        return

    for target in state['group_members'][group_id]:
        if target == idx or state['stamp'][target] < 0:
            continue
        if abs(owner_stamp - state['stamp'][target]) > int(gap_limit):
            continue
        _upsert_buffer_entry(
            state['client_buffers'][target],
            owner_idx=idx,
            owner_stamp=owner_stamp,
            feature=feature,
            priority_alpha=priority_alpha,
        )

    _prune_similarity_buffers(state, args)


def _kmeans_np(features, num_clusters, num_iters):
    num_points = features.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.int64)

    num_clusters = max(1, min(int(num_clusters), num_points))
    if num_clusters == 1:
        return np.zeros(num_points, dtype=np.int64)

    init_idx = np.random.choice(num_points, num_clusters, replace=False)
    centroids = features[init_idx].copy()

    for _ in range(int(num_iters)):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)

        new_centroids = []
        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            if not np.any(mask):
                ridx = np.random.randint(0, num_points)
                new_centroids.append(features[ridx].copy())
            else:
                new_centroids.append(features[mask].mean(axis=0))
        centroids = np.stack(new_centroids, axis=0)

    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def _spectral_partition(state, members, max_splits, args):
    if len(members) <= 1 or max_splits <= 1:
        return [0] * len(members), 0.0

    sim = state['similarity'][members][:, members].detach().cpu().numpy()
    sim = np.maximum(sim, 0.0)
    np.fill_diagonal(sim, 1.0)

    degree = sim.sum(axis=1)
    degree[degree <= 1e-12] = 1.0
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(len(members), dtype=np.float64) - inv_sqrt @ sim @ inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigengap_topk = max(2, int(_getattr(args, 'casa_eigengap_topk', 10)))
    max_candidate = min(int(max_splits), eigengap_topk, len(members) - 1)
    if max_candidate <= 0:
        return [0] * len(members), 0.0

    gaps = eigvals[1:max_candidate + 1] - eigvals[:max_candidate]
    if gaps.size == 0:
        return [0] * len(members), 0.0

    best = int(np.argmax(gaps))
    gap = float(gaps[best])
    num_parts = best + 1
    if num_parts <= 1 or gap <= 1e-12:
        return [0] * len(members), gap

    features = eigvecs[:, :num_parts]
    row_norm = np.linalg.norm(features, axis=1, keepdims=True)
    row_norm[row_norm <= 1e-12] = 1.0
    features = features / row_norm

    assignments = _kmeans_np(
        features,
        num_clusters=num_parts,
        num_iters=_getattr(args, 'casa_kmeans_iters', 20),
    )
    return assignments.tolist(), gap


def _rebuild_group_state(state, groups, args):
    groups = [sorted(group) for group in groups if group]
    if not groups:
        groups = [list(range(int(args.num_users)))]

    state['num_groups'] = len(groups)
    state['group_members'] = groups
    state['group_ids'] = [-1] * int(args.num_users)
    state['group_alpha'] = []

    for gid, members in enumerate(groups):
        alpha_c = _cluster_decay(state['iterations'], len(members), args)
        state['group_alpha'].append(alpha_c)
        for idx in members:
            state['group_ids'][idx] = gid


def _maybe_recluster(state, args, force=False):
    interval = max(1, int(_getattr(args, 'casa_cluster_interval', 1)))
    if not force and state['iterations'] % interval != 0:
        return

    current_groups = [group for group in state['group_members'] if group]
    if not current_groups:
        current_groups = [list(range(int(args.num_users)))]

    max_groups = max(1, int(_getattr(args, 'casa_num_groups', 1)))
    min_cluster = max(2, int(_getattr(args, 'casa_min_cluster_size', 4)))
    max_splits = max(2, int(_getattr(args, 'casa_max_splits', 4)))
    gamma = float(_getattr(args, 'casa_partition_gamma', 1.0))

    next_groups = []
    remaining_groups = len(current_groups)

    for members in current_groups:
        remaining_groups -= 1
        available = max_groups - len(next_groups) - remaining_groups
        allowed_parts = min(len(members), max_splits, max(1, available))

        if len(members) < min_cluster or allowed_parts <= 1:
            next_groups.append(members)
            continue

        assignments, gap = _spectral_partition(state, members, allowed_parts, args)
        alpha_c = _cluster_decay(state['iterations'], len(members), args)
        num_parts = max(assignments) + 1 if assignments else 1

        if (not force and alpha_c >= gamma * gap) or num_parts <= 1:
            next_groups.append(members)
            continue

        split_groups = [[] for _ in range(num_parts)]
        for local_pos, cluster_id in enumerate(assignments):
            split_groups[int(cluster_id)].append(members[local_pos])

        split_groups = [group for group in split_groups if group]
        if len(split_groups) <= 1:
            next_groups.append(members)
            continue

        next_groups.extend(split_groups)

    _rebuild_group_state(state, next_groups[:max_groups], args)


def init_state(state, args, random_cost):
    state['cache_device'] = resolve_cache_device(args)
    num_users = int(args.num_users)

    state['delta'] = [sd_cache_zero_like(state['w_glob'], args) for _ in range(num_users)]
    state['stamp'] = [0] * num_users
    state['cost'] = [-1] * num_users
    state['delays'] = [0] * num_users
    state['iterations'] = 0
    state['global_cost'] = 0

    state['sketch_plan'] = _build_sketch_plan(
        state['w_glob'],
        _getattr(args, 'casa_sketch_dim', 256),
    )
    state['client_features'] = [None] * num_users
    state['client_buffers'] = [[] for _ in range(num_users)]

    feature_list = []
    for idx in range(num_users):
        delta = _local_train_delta(state, args, idx)
        feature = _encode_delta(delta, state['sketch_plan'])
        state['delta'][idx] = sd_to_cache(delta, args)
        state['client_features'][idx] = feature
        feature_list.append(feature)

    state['similarity'] = _compute_similarity_matrix(feature_list)
    state['similarity_round'] = np.zeros((num_users, num_users), dtype=np.int64)
    state['group_members'] = [list(range(num_users))]
    state['group_ids'] = [0] * num_users
    state['group_alpha'] = [_cluster_decay(0, num_users, args)]

    _maybe_recluster(state, args, force=True)

    sampled_idx = random.sample(range(num_users), min(int(args.concurrency), num_users))
    for idx in sampled_idx:
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, random_cost):
    active_list = [i for i, cost in enumerate(state['cost']) if cost > 0]
    buffer_list = _select_buffer(state['cost'], args.buffer_size)
    if not buffer_list:
        return state

    state['global_cost'] = state['cost'][buffer_list[-1]]
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    client_infos = []
    for idx in buffer_list:
        state['cost'][idx] = -1
        staleness = max(0, state['iterations'] - state['stamp'][idx])
        state['delays'][idx] = staleness

        group_id = state['group_ids'][idx]
        if group_id < 0 or group_id >= len(state['group_members']):
            group_size = int(args.num_users)
        else:
            group_size = max(1, len(state['group_members'][group_id]))

        alpha_i, alpha_c, _ = _client_decay(
            state['iterations'],
            group_size,
            staleness,
            args,
        )
        client_infos.append({
            'idx': idx,
            'alpha_i': alpha_i,
            'alpha_c': alpha_c,
        })

    for info in client_infos:
        _consume_similarity_buffer(state, info['idx'])
        _insert_arrival_into_buffers(state, args, info['idx'], info['alpha_i'])

    raw_weights = [info['alpha_i'] for info in client_infos]
    total_weight = sum(raw_weights)
    if total_weight <= 1e-12:
        weights = [1.0 / len(client_infos)] * len(client_infos)
    else:
        weights = [weight / total_weight for weight in raw_weights]

    aggregated_diff = sd_zero_like(state['w_glob'])
    for weight, info in zip(weights, client_infos):
        sd_axpy(aggregated_diff, weight, state['delta'][info['idx']])

    step_size = float(args.global_lr) if args.global_lr is not None else 1.0
    with torch.no_grad():
        for key in aggregated_diff.keys():
            state['w_glob'][key].add_(aggregated_diff[key], alpha=step_size)
    load_param_dict_(state['net_glob'], state['w_glob'])

    state['iterations'] += 1
    _maybe_recluster(state, args, force=False)

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(buffer_list)]

    for idx in sampled_idx:
        delta = _local_train_delta(state, args, idx)
        feature = _encode_delta(delta, state['sketch_plan'])

        state['delta'][idx] = sd_to_cache(delta, args)
        state['client_features'][idx] = feature
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)
        state['similarity'][idx, idx] = 1.0

    return state

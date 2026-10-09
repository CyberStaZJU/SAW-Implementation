#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Subset


def extract_labels(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)

    if hasattr(dataset, 'labels'):
        return np.array(dataset.labels)

    if hasattr(dataset, 'samples'):
        return np.array([s[1] for s in dataset.samples])

    if isinstance(dataset, Subset):
        base_labels = extract_labels(dataset.dataset)
        return base_labels[np.array(dataset.indices)]

    print('Warning: extracting labels via __getitem__, this may be slow.')
    return np.array([dataset[i][1] for i in range(len(dataset))])


def dirichlet_distribution_noniid_slice(labels, num_users, alpha):
    classes = len(np.unique(labels))
    size, min_size = 0, 1

    while size < min_size:
        idx_slice = [[] for _ in range(num_users)]
        for k in range(classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            prop = np.random.dirichlet(np.repeat(alpha, num_users))
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]

            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]

        size = min(len(idx_j) for idx_j in idx_slice)

    for i in range(num_users):
        np.random.shuffle(idx_slice[i])

    return idx_slice


def generate_iid(dataset, num_users):
    l = len(dataset)
    idxs = np.arange(l)
    np.random.shuffle(idxs)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    shard = l // num_users

    for i in range(num_users):
        start = i * shard
        end = (i + 1) * shard if i != num_users - 1 else l
        dict_users[i] = idxs[start:end]

    return dict_users


def generate_noniid(dataset, num_users, alpha):
    labels = extract_labels(dataset)
    idxs = dirichlet_distribution_noniid_slice(labels, num_users, alpha)
    dict_users = {i: np.array(idxs[i], dtype='int64') for i in range(num_users)}
    return dict_users


def generate_label_block(dataset, num_users, num_groups=5, seed=None, return_metadata=False):
    """Split clients into label-disjoint groups.

    Each client group owns one contiguous block of labels and receives no
    samples from other blocks. Within a block, samples are shuffled and split
    IID across the clients assigned to that block.
    """
    num_users = int(num_users)
    num_groups = int(num_groups)
    if num_users <= 0:
        raise ValueError("num_users must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_users < num_groups:
        raise ValueError("num_users must be at least num_groups for label_block")

    labels = extract_labels(dataset)
    classes = np.sort(np.unique(labels))
    if len(classes) < num_groups:
        raise ValueError(
            f"label_block requires at least {num_groups} labels, found {len(classes)}"
        )

    rng = np.random.default_rng(seed)
    label_groups = [group.astype(int).tolist() for group in np.array_split(classes, num_groups)]

    base_users, user_remainder = divmod(num_users, num_groups)
    users_per_group = [base_users + (1 if group_id < user_remainder else 0)
                       for group_id in range(num_groups)]

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    client_group_ids = np.zeros(num_users, dtype=np.int64)
    user_cursor = 0
    for group_id, group_labels in enumerate(label_groups):
        group_labels = np.asarray(group_labels, dtype=labels.dtype)
        group_indices = np.where(np.isin(labels, group_labels))[0]
        rng.shuffle(group_indices)

        group_user_count = users_per_group[group_id]
        splits = np.array_split(group_indices, group_user_count)
        for local_user, local_indices in enumerate(splits):
            user_idx = user_cursor + local_user
            dict_users[user_idx] = np.asarray(local_indices, dtype='int64')
            client_group_ids[user_idx] = group_id
        user_cursor += group_user_count

    if not return_metadata:
        return dict_users

    metadata = {
        "num_groups": num_groups,
        "label_groups": label_groups,
        "client_group_ids": client_group_ids.astype(int).tolist(),
        "users_per_group": users_per_group,
    }
    return dict_users, metadata


def generate_label_correlated(
    dataset,
    num_users,
    num_groups=5,
    primary_fraction=0.5,
    seed=None,
    return_metadata=False,
):
    """Split clients into groups with soft label-block dominance.

    Each sample is assigned to its own label block's client group with
    probability ``primary_fraction``; otherwise it is assigned to one of the
    other groups. This keeps label and client-group identity correlated while
    avoiding hard label exclusivity.
    """
    num_users = int(num_users)
    num_groups = int(num_groups)
    primary_fraction = float(primary_fraction)
    if num_users <= 0:
        raise ValueError("num_users must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_users < num_groups:
        raise ValueError("num_users must be at least num_groups for label_correlated")
    if not 0.0 <= primary_fraction <= 1.0:
        raise ValueError("label_primary_fraction must be in [0, 1]")

    labels = extract_labels(dataset)
    classes = np.sort(np.unique(labels))
    if len(classes) < num_groups:
        raise ValueError(
            f"label_correlated requires at least {num_groups} labels, found {len(classes)}"
        )

    rng = np.random.default_rng(seed)
    label_groups = [group.astype(int).tolist() for group in np.array_split(classes, num_groups)]
    label_to_group = {
        int(label): group_id
        for group_id, group_labels in enumerate(label_groups)
        for label in group_labels
    }

    base_users, user_remainder = divmod(num_users, num_groups)
    users_per_group = [base_users + (1 if group_id < user_remainder else 0)
                       for group_id in range(num_groups)]

    group_pools = [[] for _ in range(num_groups)]
    for sample_idx, label in enumerate(labels):
        primary_group = label_to_group[int(label)]
        if rng.random() < primary_fraction or num_groups == 1:
            target_group = primary_group
        else:
            alternatives = [gid for gid in range(num_groups) if gid != primary_group]
            target_group = int(rng.choice(alternatives))
        group_pools[target_group].append(int(sample_idx))

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    client_group_ids = np.zeros(num_users, dtype=np.int64)
    user_cursor = 0
    for group_id, group_indices in enumerate(group_pools):
        group_indices = np.asarray(group_indices, dtype=np.int64)
        rng.shuffle(group_indices)
        group_user_count = users_per_group[group_id]
        splits = np.array_split(group_indices, group_user_count)
        for local_user, local_indices in enumerate(splits):
            user_idx = user_cursor + local_user
            dict_users[user_idx] = np.asarray(local_indices, dtype='int64')
            client_group_ids[user_idx] = group_id
        user_cursor += group_user_count

    if not return_metadata:
        return dict_users

    metadata = {
        "num_groups": num_groups,
        "label_groups": label_groups,
        "client_group_ids": client_group_ids.astype(int).tolist(),
        "users_per_group": users_per_group,
        "primary_fraction": primary_fraction,
    }
    return dict_users, metadata


def _dirichlet_partition_indices(labels, num_users, alpha, rng):
    classes = np.sort(np.unique(labels))
    size, min_size = 0, 1

    while size < min_size:
        idx_slice = [[] for _ in range(num_users)]
        for label in classes:
            idx_k = np.where(labels == label)[0]
            rng.shuffle(idx_k)

            prop = rng.dirichlet(np.repeat(float(alpha), num_users))
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]

            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]

        size = min(len(idx_j) for idx_j in idx_slice)

    for idx_j in idx_slice:
        rng.shuffle(idx_j)
    return idx_slice


def generate_speed_label_correlated_dirichlet(
    dataset,
    num_users,
    speed_group_ids,
    alpha=0.1,
    primary_fraction=0.9,
    seed=None,
    return_metadata=False,
):
    """Bind label blocks to fixed speed tiers, then split within each tier by Dirichlet.

    This creates a harder setting than plain label_correlated: the label direction
    is correlated with the hierarchical speed group, and clients inside the same
    speed tier remain strongly non-IID through a Dirichlet partition.
    """
    num_users = int(num_users)
    alpha = float(alpha)
    primary_fraction = float(primary_fraction)
    if num_users <= 0:
        raise ValueError("num_users must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= primary_fraction <= 1.0:
        raise ValueError("label_primary_fraction must be in [0, 1]")
    if len(speed_group_ids) != num_users:
        raise ValueError("speed_group_ids length must match num_users")

    labels = extract_labels(dataset)
    classes = np.sort(np.unique(labels))
    canonical_speed_order = ['Small', 'Medium', 'Large', 'BlockFast', 'BlockMidFast', 'BlockMedium', 'BlockSlow', 'BlockVerySlow']
    speed_group_set = set(speed_group_ids)
    speed_categories = [speed for speed in canonical_speed_order if speed in speed_group_set]
    speed_categories.extend([speed for speed in dict.fromkeys(speed_group_ids) if speed not in set(speed_categories)])
    num_groups = len(speed_categories)
    if len(classes) < num_groups:
        raise ValueError(
            f"speed_label_correlated_dirichlet requires at least {num_groups} labels, "
            f"found {len(classes)}"
        )

    rng = np.random.default_rng(seed)
    label_groups = [group.astype(int).tolist() for group in np.array_split(classes, num_groups)]
    label_to_group = {
        int(label): group_id
        for group_id, group_labels in enumerate(label_groups)
        for label in group_labels
    }

    group_pools = [[] for _ in range(num_groups)]
    for sample_idx, label in enumerate(labels):
        primary_group = label_to_group[int(label)]
        if rng.random() < primary_fraction or num_groups == 1:
            target_group = primary_group
        else:
            alternatives = [gid for gid in range(num_groups) if gid != primary_group]
            target_group = int(rng.choice(alternatives))
        group_pools[target_group].append(int(sample_idx))

    users_by_group = [[] for _ in range(num_groups)]
    speed_to_group = {speed: group_id for group_id, speed in enumerate(speed_categories)}
    client_group_ids = np.zeros(num_users, dtype=np.int64)
    for user_idx, speed in enumerate(speed_group_ids):
        group_id = speed_to_group[speed]
        users_by_group[group_id].append(int(user_idx))
        client_group_ids[user_idx] = group_id

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for group_id, user_ids in enumerate(users_by_group):
        if not user_ids:
            continue

        group_indices = np.asarray(group_pools[group_id], dtype=np.int64)
        rng.shuffle(group_indices)
        if group_indices.size == 0:
            continue

        group_labels = labels[group_indices]
        local_slices = _dirichlet_partition_indices(
            group_labels,
            len(user_ids),
            alpha,
            rng,
        )
        for local_user_pos, local_indices in enumerate(local_slices):
            user_idx = user_ids[local_user_pos]
            dict_users[user_idx] = group_indices[np.asarray(local_indices, dtype=np.int64)]

    if not return_metadata:
        return dict_users

    metadata = {
        "num_groups": num_groups,
        "label_groups": label_groups,
        "client_group_ids": client_group_ids.astype(int).tolist(),
        "users_per_group": [len(user_ids) for user_ids in users_by_group],
        "speed_categories": speed_categories,
        "speed_group_ids": list(speed_group_ids),
        "primary_fraction": primary_fraction,
        "alpha": alpha,
    }
    return dict_users, metadata


def generate_inter_noniid(dataset, num_users, num_groups, alpha):
    assert num_users % num_groups == 0, "num_users must be divisible by num_groups"

    users_per_group = num_users // num_groups
    labels = extract_labels(dataset)
    group_idxs = dirichlet_distribution_noniid_slice(labels, num_groups, alpha)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for group_idx, group_data_idxs in enumerate(group_idxs):
        group_data_idxs = np.array(group_data_idxs)
        local_idxs = np.arange(len(group_data_idxs))
        np.random.shuffle(local_idxs)

        shard = len(group_data_idxs) // users_per_group
        for user_in_group in range(users_per_group):
            user_idx = group_idx * users_per_group + user_in_group
            start = user_in_group * shard
            end = (user_in_group + 1) * shard if user_in_group != users_per_group - 1 else len(group_data_idxs)
            dict_users[user_idx] = group_data_idxs[local_idxs[start:end]].astype('int64')

    return dict_users


def generate_intra_noniid(dataset, num_users, num_groups, alpha):
    assert num_users % num_groups == 0, "num_users must be divisible by num_groups"

    users_per_group = num_users // num_groups
    group_idxs = generate_iid(dataset, num_groups)
    all_labels = extract_labels(dataset)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for group_idx, group_data_idxs in group_idxs.items():
        group_data_idxs = np.array(group_data_idxs)
        group_labels = all_labels[group_data_idxs]

        local_user_idxs = dirichlet_distribution_noniid_slice(
            group_labels, users_per_group, alpha
        )

        for user_in_group, local_idxs in enumerate(local_user_idxs):
            user_idx = group_idx * users_per_group + user_in_group
            dict_users[user_idx] = group_data_idxs[np.array(local_idxs)].astype('int64')

    return dict_users

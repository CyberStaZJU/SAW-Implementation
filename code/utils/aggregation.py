import torch
try:
    import torch_npu
except ImportError:
    pass
import numpy as np

DELAY_CATEGORIES = ('Small', 'Medium', 'Large')
DELAY_PROFILE_RANGES = {
    'hierarchical': {
        'Small': (1.0, 2.0),
        'Medium': (3.0, 5.0),
        'Large': (5.0, 8.0),
    },
    'label_correlated_hierarchical': {
        'BlockFast': (1.0, 2.0),
        'BlockMidFast': (2.0, 4.0),
        'BlockMedium': (4.0, 7.0),
        'BlockSlow': (7.0, 11.0),
        'BlockVerySlow': (10.0, 16.0),
    },
    'mild_label_correlated_hierarchical': {
        'Small': (1.0, 2.0),
        'Medium': (3.0, 5.0),
        'Large': (5.0, 8.0),
    },
}

DELAY_PROFILE_MIXTURES = {
    'label_correlated_hierarchical': {
        # Slow groups are slow on average, but occasionally return quickly so
        # their caches can still be refreshed.
        'BlockSlow': (
            (0.15, (1.0, 2.5)),
            (0.85, (7.0, 11.0)),
        ),
        'BlockVerySlow': (
            (0.10, (1.0, 3.0)),
            (0.90, (10.0, 16.0)),
        ),
    },
}


def normalize_delay_profile(profile):
    return str(profile)


def build_client_delay_profile(num_users, gamma=1.0, seed=None):
    """Assign each client a fixed, balanced Small/Medium/Large delay label."""
    num_users = int(num_users)
    if num_users < 0:
        raise ValueError("num_users must be non-negative")
    if num_users == 0:
        return [], []

    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("delay_gamma must be positive")

    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(num_users, len(DELAY_CATEGORIES))
    counts = [base_count] * len(DELAY_CATEGORIES)
    for idx in range(remainder):
        counts[idx] += 1

    groups = [
        category
        for category, count in zip(DELAY_CATEGORIES, counts)
        for _ in range(count)
    ]
    rng.shuffle(groups)
    category_probs = [count / num_users for count in counts]
    return groups, category_probs


def sample_client_delay(client_delay_groups, idx, profile='hierarchical', rng=None):
    """Sample one wall-clock time from a client's fixed delay category."""
    profile = normalize_delay_profile(profile)
    if profile not in DELAY_PROFILE_RANGES:
        raise ValueError(f"Unsupported delay profile: {profile}")
    group = client_delay_groups[int(idx)]
    if rng is None:
        rng = np.random.default_rng()

    mixture = DELAY_PROFILE_MIXTURES.get(profile, {}).get(group)
    if mixture:
        probs = np.array([float(prob) for prob, _range in mixture], dtype=np.float64)
        probs = probs / probs.sum()
        choice = int(rng.choice(len(mixture), p=probs))
        low, high = mixture[choice][1]
        return float(rng.uniform(low, high))

    low, high = DELAY_PROFILE_RANGES[profile][group]
    return float(rng.uniform(low, high))


def random_cost(distribution='pareto'):
    distribution = normalize_delay_profile(distribution)
    if distribution == 'pareto':
        return 10 * np.random.pareto(2)
    if distribution == 'gauss':
        return np.abs(np.random.normal(scale=12.5))
    raise ValueError(f"Unsupported random_cost distribution: {distribution}")

def rescaling(delay, buffer_list):
    weights = [1.0 / (d + 1) if d is not None else 0 for d in delay]
    masked = [
        w if (d is not None) or (i in buffer_list) else 0
        for i, (d, w) in enumerate(zip(delay, weights))
    ]
    total_weight = sum(masked)
    return [w / total_weight for w in masked]

def buffered_aggregation(weight, Delta, ref_model):
    if not Delta:
        raise ValueError("Delta cannot be empty")

    keys = None
    for d in Delta:
        if d is not None:
            keys = [k for k in d.keys()]
            break
    if keys is None:
        raise ValueError("All Delta entries are None")

    w_avg = {k: torch.zeros_like(ref_model[k]) for k in keys}
    for k in keys:
        for i in range(len(weight)):
            if weight[i] > 0 and Delta[i] is not None:
                value = Delta[i].get(k)
                if value is None:
                    value = torch.zeros_like(w_avg[k])
                w_avg[k] += weight[i] * value.to(w_avg[k].device)
    return w_avg

# algorithms/algo_dispatcher.py
import numpy as np

from utils.aggregation import (
    DELAY_CATEGORIES,
    build_client_delay_profile,
    normalize_delay_profile,
    random_cost,
    sample_client_delay,
)
from utils.fedscale_trace import ensure_fedscale_trace_sampler
from algorithm import adamasfl, fedasync, fedbuff, fedbuffma, saw, ca2fl, casa, fadas, saw_equalw, saw_adainside, arrival_frozen_cache, saw_cache_ablation

FADAS_UNIFORM_COSTS = {
    'hierarchical',
    'label_correlated_hierarchical',
    'mild_label_correlated_hierarchical',
}
FEDSCALE_TRACE_COSTS = {
    'fedscale_trace',
}
LABEL_CORRELATED_SPEED_CATEGORIES = (
    'BlockFast',
    'BlockMidFast',
    'BlockMedium',
    'BlockSlow',
    'BlockVerySlow',
)
MILD_LABEL_CORRELATED_SPEED_CATEGORIES = DELAY_CATEGORIES


def _is_fadas_uniform(args):
    return normalize_delay_profile(getattr(args, 'random_cost', 'pareto')) in FADAS_UNIFORM_COSTS


def _is_label_correlated_hierarchical(args):
    return normalize_delay_profile(getattr(args, 'random_cost', 'pareto')) in {
        'label_correlated_hierarchical',
        'mild_label_correlated_hierarchical',
    }


def _is_fedscale_trace(args):
    return normalize_delay_profile(getattr(args, 'random_cost', 'pareto')) in FEDSCALE_TRACE_COSTS


def _build_label_correlated_hierarchical_delay_profile(state, args, seed):
    if state is None:
        raise ValueError("label_correlated_hierarchical delay requires dataset state")

    attr_prefix = normalize_delay_profile(getattr(args, 'random_cost', 'label_correlated_hierarchical'))
    group_ids = getattr(args, 'label_correlated_group_ids', None)
    num_blocks = int(getattr(args, 'label_correlated_num_groups', 0) or 0)

    if group_ids is None:
        raise ValueError(
            f"{attr_prefix} delay requires label-derived client groups; "
            "use --distribution label_correlated or --distribution noniid"
        )

    group_ids = np.asarray(group_ids, dtype=np.int64)
    num_users = int(args.num_users)
    if group_ids.size != num_users:
        raise ValueError(
            f"label_correlated_hierarchical expected {num_users} client group ids, "
            f"found {group_ids.size}"
        )

    if num_blocks <= 0:
        num_blocks = int(group_ids.max()) + 1
    speed_categories = (
        MILD_LABEL_CORRELATED_SPEED_CATEGORIES
        if attr_prefix == 'mild_label_correlated_hierarchical'
        else LABEL_CORRELATED_SPEED_CATEGORIES
    )
    if num_blocks != len(speed_categories):
        raise ValueError(
            f"{attr_prefix} currently expects exactly {len(speed_categories)} label groups; "
            f"found {num_blocks}"
        )

    rng = np.random.default_rng(int(seed) + 104729)
    speed_order = rng.permutation(num_blocks)
    block_to_speed = {
        int(block_id): speed_categories[rank]
        for rank, block_id in enumerate(speed_order)
    }
    delay_groups = [block_to_speed[int(group_id)] for group_id in group_ids]
    speed_counts = [delay_groups.count(category) for category in speed_categories]
    probs = [count / float(max(1, num_users)) for count in speed_counts]

    block_counts = np.bincount(group_ids, minlength=num_blocks)
    setattr(args, f'{attr_prefix}_direction_group_ids', group_ids.astype(int).tolist())
    setattr(args, f'{attr_prefix}_direction_group_counts', block_counts.astype(int).tolist())
    setattr(args, f'{attr_prefix}_speed_order', speed_order.astype(int).tolist())
    setattr(args, f'{attr_prefix}_block_to_speed', {
        int(block_id): speed_name for block_id, speed_name in block_to_speed.items()
    })
    return delay_groups, probs


def _ensure_client_delay_profile(args, state=None):
    num_users = int(args.num_users)
    groups = getattr(args, 'client_delay_groups', None)
    if groups is not None and len(groups) == num_users:
        if not hasattr(args, 'client_delay_rng'):
            seed = getattr(args, 'delay_seed', None)
            if seed is None:
                seed = int(getattr(args, 'seed', 0))
            args.client_delay_rng = np.random.default_rng(seed + 1)
        return groups

    seed = getattr(args, 'delay_seed', None)
    if seed is None:
        seed = int(getattr(args, 'seed', 0))

    if _is_label_correlated_hierarchical(args):
        groups, probs = _build_label_correlated_hierarchical_delay_profile(state, args, seed)
    else:
        groups, probs = build_client_delay_profile(
            num_users=num_users,
            gamma=getattr(args, 'delay_gamma', 1.0),
            seed=seed,
        )
    args.client_delay_groups = groups
    args.client_delay_group_probs = probs
    args.client_delay_rng = np.random.default_rng(seed + 1)
    return groups


def _paper_uniform_cost(args, idx):
    if idx is None:
        raise ValueError("client-indexed delay profile requires a client index")
    groups = _ensure_client_delay_profile(args)
    return sample_client_delay(
        groups,
        idx,
        profile=getattr(args, 'random_cost', 'hierarchical'),
        rng=args.client_delay_rng,
    )


def _attach_delay_metadata(state, args):
    if _is_fedscale_trace(args):
        ensure_fedscale_trace_sampler(args, state)
        return

    if not _is_fadas_uniform(args):
        return
    _ensure_client_delay_profile(args, state)
    state['client_delay_groups'] = list(args.client_delay_groups)
    state['client_delay_group_probs'] = list(args.client_delay_group_probs)
    attr_prefix = normalize_delay_profile(getattr(args, 'random_cost', 'pareto'))
    direction_group_attr = f'{attr_prefix}_direction_group_ids'
    if hasattr(args, direction_group_attr):
        state[direction_group_attr] = list(getattr(args, direction_group_attr))
        state[f'{attr_prefix}_direction_group_counts'] = list(
            getattr(args, f'{attr_prefix}_direction_group_counts')
        )
        speed_order_attr = f'{attr_prefix}_speed_order'
        block_to_speed_attr = f'{attr_prefix}_block_to_speed'
        if hasattr(args, speed_order_attr):
            state[speed_order_attr] = list(getattr(args, speed_order_attr))
        if hasattr(args, block_to_speed_attr):
            state[block_to_speed_attr] = dict(getattr(args, block_to_speed_attr))


def _random_cost(args, state=None):
    if _is_fedscale_trace(args):
        sampler = ensure_fedscale_trace_sampler(args, state)
        return lambda idx=None: sampler.sample_client_delay(
            idx,
            float(state.get('global_cost', 0.0)) if state is not None else 0.0,
        )
    if _is_fadas_uniform(args):
        _ensure_client_delay_profile(args, state)
        return lambda idx=None: _paper_uniform_cost(args, idx)
    return lambda idx=None: random_cost(getattr(args, 'random_cost', 'pareto'))

def init_state(state, args):
    random_cost = _random_cost(args, state)
    _attach_delay_metadata(state, args)
    if args.algo == 'FedBuff':
        return fedbuff.init_state(
            state, args, random_cost
        )
    elif args.algo == 'FADAS':
        return fadas.init_state(
            state, args, random_cost
        )
    elif args.algo == 'FedAsync':
        return fedasync.init_state(
            state, args, random_cost
        )
    elif args.algo == 'FedBuffMALight':
        return fedbuffma.init_state(
            state, args, random_cost
        )
    elif args.algo == 'SAW':
        return saw.init_state(
            state, args, random_cost
        )
    elif args.algo == 'SAW_EqualW':
        return saw_equalw.init_state(
            state, args, random_cost
        )
    elif args.algo == 'SAW_AdaInside':
        return saw_adainside.init_state(
            state, args, random_cost
        )
    elif args.algo == 'ArrivalFrozenCache':
        return arrival_frozen_cache.init_state(
            state, args, random_cost
        )
    elif args.algo in saw_cache_ablation.SAW_CACHE_ABLATION_ALGOS:
        return saw_cache_ablation.init_state(
            state, args, random_cost
        )
    elif args.algo == 'AdaMasFL':
        return adamasfl.init_state(state, args, random_cost)
    elif args.algo == 'CA2FL':
        return ca2fl.init_state(
            state, args, random_cost
        )

    elif args.algo == 'CASA':
        return casa.init_state(
            state, args, random_cost
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")


def run_one_round(args, state, dataset_train, dict_users, num_samples):
    random_cost = _random_cost(args, state)
    if args.algo == 'FedBuff':
        return fedbuff.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )
    
    elif args.algo == 'FADAS':
        return fadas.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )
    
    elif args.algo == 'SAW':
        return saw.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo == 'SAW_EqualW':
        return saw_equalw.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo == 'SAW_AdaInside':
        return saw_adainside.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo == 'ArrivalFrozenCache':
        return arrival_frozen_cache.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo in saw_cache_ablation.SAW_CACHE_ABLATION_ALGOS:
        return saw_cache_ablation.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo == 'FedBuffMALight':
        return fedbuffma.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )

    elif args.algo == 'FedAsync':
        return fedasync.run_round(
            state, args, dataset_train, dict_users, num_samples, random_cost
        )

    elif args.algo == 'AdaMasFL':
        return adamasfl.run_round(
            state, args, dataset_train, dict_users, num_samples, random_cost
        )

    elif args.algo == 'CA2FL':
        return ca2fl.run_round(
            state, args, dataset_train, dict_users, num_samples, random_cost
        )

    elif args.algo == 'CASA':
        return casa.run_round(
            state=state,
            args=args,
            random_cost=random_cost,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys








def _is_femnist_task(args):
    return str(args.dataset).lower() == 'femnist'




def _is_gspeech_task(args):
    return str(args.dataset).lower() == 'gspeech'


def _flag_was_provided(argv, flags):
    return any(
        token == flag or token.startswith(flag + '=')
        for token in argv
        for flag in flags
    )


def _maybe_set_default(args, argv, attr, flags, value):
    if _flag_was_provided(argv, flags):
        return

    resolved = value(args) if callable(value) else value
    setattr(args, attr, resolved)


def _apply_overrides(args, argv, overrides):
    for attr, flags, value in overrides:
        _maybe_set_default(args, argv, attr, flags, value)
    return args


# ---------------------------------------------------------------------------
# Contextual defaults
# ---------------------------------------------------------------------------



def _apply_femnist_defaults(args, argv):
    if not _is_femnist_task(args):
        return args

    args.dataset = 'femnist'
    return _apply_overrides(
        args,
        argv,
        (
            ('model', ['--model'], 'cnn'),
            ('num_users', ['--num_users'], 1000),
            ('femnist_min_samples_per_client', ['--femnist_min_samples_per_client'], 250),
            ('num_classes', ['--num_classes'], 62),
            ('num_channels', ['--num_channels'], 1),
        ),
    )




def _apply_gspeech_defaults(args, argv):
    if not _is_gspeech_task(args):
        return args

    args.dataset = 'gspeech'
    return _apply_overrides(
        args,
        argv,
        (
            ('model', ['--model'], 'melcnn'),
            ('num_users', ['--num_users'], 1000),
            ('gspeech_min_samples_per_client', ['--gspeech_min_samples_per_client'], 30),
            ('num_classes', ['--num_classes'], 35),
            ('num_channels', ['--num_channels'], 1),
        ),
    )




def _apply_casa_defaults(args, argv):
    if args.algo != 'CASA':
        return args

    return _apply_overrides(
        args,
        argv,
        (
            ('casa_num_groups', ['--casa_num_groups'], 4),
            ('casa_alpha0', ['--casa_alpha0'], 0.3),
            ('casa_partition_gamma', ['--casa_partition_gamma'], 0.15),
            ('casa_cluster_bias', ['--casa_cluster_bias'], 3.0),
            ('casa_eigengap_topk', ['--casa_eigengap_topk'], 10),
            ('casa_cluster_interval', ['--casa_cluster_interval'], 5),
            ('casa_min_cluster_size', ['--casa_min_cluster_size'], 8),
            ('casa_sketch_dim', ['--casa_sketch_dim'], 128),
            (
                'casa_buffer_budget',
                ['--casa_buffer_budget'],
                lambda parsed: max(int(parsed.num_users), int(parsed.buffer_size) * 4),
            ),
            ('casa_start_gap', ['--casa_start_gap'], lambda parsed: parsed.local_period),
        ),
    )


def _apply_fedbuffma_defaults(args, argv):
    if args.algo != 'FedBuffMALight':
        return args

    _maybe_set_default(args, argv, 'ma_staleness_limit', ['--ma_staleness_limit'], 20)
    return args


def _apply_fadas_defaults(args, argv):
    if args.algo != 'FADAS':
        return args

    lr, global_lr = 0.1, 0.0001
    _maybe_set_default(args, argv, 'lr', ['--lr'], lr)
    _maybe_set_default(args, argv, 'global_lr', ['--global_lr'], global_lr)

    _maybe_set_default(args, argv, 'fadas_beta1', ['--fadas_beta1'], 0.9)
    _maybe_set_default(args, argv, 'fadas_beta2', ['--fadas_beta2'], 0.99)
    _maybe_set_default(args, argv, 'fadas_eps', ['--fadas_eps'], 1e-8)
    _maybe_set_default(
        args,
        argv,
        'fadas_tau_c',
        ['--fadas_tau_c'],
        lambda parsed: max(1.0, float(parsed.concurrency) / float(max(1, int(parsed.buffer_size)))),
    )
    return args


def _apply_fedasync_defaults(args, argv):
    if args.algo != 'FedAsync':
        return args

    _maybe_set_default(args, argv, 'fedasync_rho', ['--fedasync_rho'], 0.005)
    return args


def _apply_algo_defaults(args, argv):
    args = _apply_femnist_defaults(args, argv)
    args = _apply_gspeech_defaults(args, argv)
    if (
        not _flag_was_provided(argv, ['--distribution'])
        and not _is_femnist_task(args)
        and not _is_gspeech_task(args)
    ):
        if getattr(args, 'random_cost', '') == 'label_correlated_hierarchical':
            args.distribution = 'label_correlated'
    for apply_defaults in (
        _apply_casa_defaults,
        _apply_fedbuffma_defaults,
        _apply_fadas_defaults,
        _apply_fedasync_defaults,
    ):
        args = apply_defaults(args, argv)
    return args


# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------

def _add_misc_args(parser):
    group = parser.add_argument_group('Misc')
    group.add_argument('--momentum', type=float, default=0, help='SGD momentum')
    group.add_argument('--verbose', action='store_true', help='verbose print')
    group.add_argument('--num_workers', type=int, default=0, help='number of workers when loading data')
    group.add_argument('--output_dir', type=str, default='', help='directory for test accuracy output')
    group.add_argument('--run_tag', type=str, default='', help='optional tag appended to output filenames')
    group.add_argument('--direction_skew_log_dir', type=str, default='', help='directory for per-round direction-skew CSV logs')
    group.add_argument('--direction_skew_log_every', type=int, default=1, help='log direction-skew metrics every N server rounds')
    group.add_argument('--system_metrics_log_dir', type=str, default='', help='directory for per-round system-metrics CSV logs')
    group.add_argument('--validation_output_dir', type=str, default='', help='directory for held-out validation accuracy output')
    group.add_argument('--validation_fraction', type=float, default=0.0, help='fraction of the vision training set reserved for validation')
    group.add_argument('--validation_split_seed', type=int, default=731, help='seed for the deterministic validation split')
    group.add_argument('--direction_skew_groups', type=int, default=4, help='fixed monitor groups for direction-skew logging; currently forced to 4 balanced label-histogram groups')


def _add_federated_args(parser):
    group = parser.add_argument_group('Federated')
    group.add_argument('--total_rounds', type=int, default=500, help='rounds of training')
    group.add_argument('--num_users', type=int, default=100, help='number of users')
    group.add_argument('--local_bs', type=int, default=100, help='local batch size')
    group.add_argument('--bs', type=int, default=500, help='test batch size')
    group.add_argument('--interval', type=int, default=1, help='number of rounds between evaluations')
    group.add_argument('--local_period', type=int, default=10, help='number of local SGD steps')
    group.add_argument('--local_epochs', type=float, default=None, help='local epochs; overrides local_period when positive')
    group.add_argument('--algo', type=str, default='SAW', help='federated learning algorithm')

def _add_model_and_runtime_args(parser):
    group = parser.add_argument_group('Model and Runtime')
    group.add_argument('--model', type=str, default='resnet', help='model name')
    group.add_argument('--dataset', choices=['cifar', 'cifar100', 'tinyimagenet', 'femnist', 'gspeech'], default='cifar', help='dataset name')
    group.add_argument('--num_classes', type=int, default=10, help='number of classes')
    group.add_argument('--num_channels', type=int, default=3, help='number of image channels')
    group.add_argument('--gpu', type=str, default='0', help='GPU ID, or -1 for CPU')
    group.add_argument('--seed', type=int, default=123, help='random seed')
    group.add_argument('--cache_device', '--cache-device', choices=['cpu', 'device'], default='cpu',
                       help='persistent client tensor storage; training and aggregation use the model device')


def _add_data_args(parser):
    group = parser.add_argument_group('Data')
    group.add_argument('--data_dir', default='', help='external dataset root; defaults to SAW_DATA_ROOT or ~/SAW-data')
    group.add_argument('--saw_cache_topk', type=int, default=5, help='number of freshest updates in SAW_FreshTopK')
    group.add_argument('--saw_cache_power', type=float, default=1.0, help='inverse-age exponent in SAW_Power')
    group.add_argument('--saw_cache_exp_lambda', type=float, default=0.2, help='age decay rate in SAW_Exp')
    group.add_argument(
        '--distribution',
        type=str,
        default='noniid',
        help='iid/noniid/label_block/label_correlated',
    )
    group.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID partitioning')
    group.add_argument('--label_block_groups', type=int, default=5, help='number of label-disjoint client groups for --distribution label_block')
    group.add_argument('--label_primary_fraction', type=float, default=0.5, help='fraction of each label-correlated group drawn from its primary label block')
    group.add_argument('--concurrency', type=int, default=20, help='number of concurrent clients')
    group.add_argument('--buffer_size', type=int, default=10, help='server buffer size')
    group.add_argument('--femnist_data_dir', type=str, default='', help='FEMNIST data directory; defaults to data/femnist_truncated when available')
    group.add_argument('--femnist_min_samples_per_client', type=int, default=0, help='minimum train samples per natural FEMNIST client; 0 disables filtering')
    group.add_argument('--gspeech_min_samples_per_client', type=int, default=0, help='minimum train samples per natural GSpeech client; 0 disables filtering')
    group.add_argument(
        '--random_cost',
        type=str,
        default='gauss',
        choices=[
            'pareto',
            'gauss',
            'hierarchical',
            'label_correlated_hierarchical',
            'mild_label_correlated_hierarchical',
            'fedscale_trace',
        ],
        help='client delay distribution',
    )
    group.add_argument(
        '--delay_gamma',
        type=float,
        default=1.0,
        help='kept for compatibility; balanced Small/Medium/Large delay assignment ignores this value',
    )
    group.add_argument(
        '--delay_seed',
        type=int,
        default=None,
        help='optional seed for fixed client delay groups; defaults to the experiment seed',
    )
    group.add_argument(
        '--fedscale_client_profile_path',
        type=str,
        default='',
        help='FedScale client profile path used by --random_cost fedscale_trace',
    )
    group.add_argument(
        '--fedscale_availability_trace_path',
        type=str,
        default='',
        help='optional FedScale client availability trace path for --random_cost fedscale_trace',
    )
    group.add_argument(
        '--fedscale_time_scale',
        type=float,
        default=1.0,
        help='multiplier applied to FedScale profile/trace times',
    )
    group.add_argument(
        '--fedscale_default_duration',
        type=float,
        default=1.0,
        help='fallback client duration when a FedScale profile record lacks usable timing fields',
    )
    group.add_argument(
        '--fedscale_min_duration',
        type=float,
        default=1e-6,
        help='minimum positive duration for FedScale trace scheduling',
    )
    group.add_argument(
        '--fedscale_upload_size_mb',
        type=float,
        default=1.0,
        help='upload size in MB used by FedScale communication latency',
    )
    group.add_argument(
        '--fedscale_download_size_mb',
        type=float,
        default=None,
        help='download size in MB used by FedScale communication latency; defaults to upload size',
    )
    group.add_argument(
        '--fedscale_batch_size',
        type=int,
        default=0,
        help='batch size used by FedScale computation latency; defaults to --local_bs',
    )
    group.add_argument(
        '--fedscale_local_steps',
        type=int,
        default=0,
        help='local steps used by FedScale computation latency; defaults to --local_period',
    )
    group.add_argument(
        '--fedscale_augmentation_factor',
        type=float,
        default=3.0,
        help='FedScale backward-pass multiplier for computation latency',
    )
    group.add_argument(
        '--fedscale_profile_sample',
        type=str,
        default='random',
        choices=['random', 'sorted'],
        help='how to map experiment client ids onto a larger FedScale profile',
    )
    group.add_argument(
        '--fedscale_no_trace_wrap',
        action='store_true',
        help='do not wrap finite FedScale availability traces cyclically',
    )
    group.add_argument(
        '--fedscale_trace_exhausted_penalty',
        type=float,
        default=3600.0,
        help='wait penalty when a non-wrapped FedScale availability trace is exhausted',
    )


def _add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--lr', type=float, default=0.01, help='local learning rate')
    group.add_argument('--global_lr', type=float, default=None, help='server/global learning rate')
    group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay used by local SGD')


def _add_momentum_approximation_args(parser):
    group = parser.add_argument_group('Momentum Approximation')
    group.add_argument('--ma_p', type=float, default=0, help='staleness down-scaling exponent')
    group.add_argument('--ma_staleness_limit', type=int, default=None, help='maximum tolerated staleness; negative disables dropping')


def _add_casa_args(parser):
    group = parser.add_argument_group('CASA')
    group.add_argument('--casa_num_groups', type=int, default=4, help='maximum number of CASA clusters')
    group.add_argument('--casa_alpha0', type=float, default=1.0, help='initial cluster-level decay')
    group.add_argument('--casa_omega', type=float, default=0.01, help='exponential decay factor in CASA Omega(t)')
    group.add_argument('--casa_buffer_budget', type=int, default=None, help='total similarity-buffer budget')
    group.add_argument('--casa_start_gap', type=int, default=None, help='maximum start-round gap allowed by CASA buffer insertion')
    group.add_argument('--casa_partition_gamma', type=float, default=1.0, help='eigengap scaling for CASA repartition')
    group.add_argument('--casa_max_splits', type=int, default=4, help='maximum sub-clusters from one CASA split')
    group.add_argument('--casa_min_cluster_size', type=int, default=4, help='minimum cluster size eligible for CASA repartition')
    group.add_argument('--casa_cluster_interval', type=int, default=1, help='round interval for CASA repartition checks')
    group.add_argument('--casa_sketch_dim', type=int, default=256, help='low-dimensional sketch size for CASA similarity tracking')
    group.add_argument('--casa_kmeans_iters', type=int, default=20, help='number of k-means iterations in CASA spectral repartition')
    group.add_argument('--casa_cluster_bias', type=float, default=3.0, help='cluster-size bias added in CASA decay denominator')
    group.add_argument('--casa_eigengap_topk', type=int, default=10, help='number of smallest eigenvalues considered by CASA')


def _add_fadas_args(parser):
    group = parser.add_argument_group('FADAS')
    group.add_argument('--fadas_beta1', type=float, default=0.9, help='first-moment coefficient')
    group.add_argument('--fadas_beta2', type=float, default=0.99, help='second-moment coefficient')
    group.add_argument('--fadas_eps', type=float, default=1e-8, help='numerical epsilon')
    group.add_argument('--fadas_tau_c', type=float, default=None, help='delay threshold for delay-adaptive FADAS')
    group.add_argument('--fadas_delay_adaptive', action='store_true', help='enable delay-adaptive FADAS learning rate')


def _add_fedasync_args(parser):
    group = parser.add_argument_group('FedAsync')
    group.add_argument('--fedasync_alpha', type=float, default=0.2, help='FedAsync base mixing weight')
    group.add_argument('--fedasync_rho', type=float, default=0.1, help='FedAsync proximal regularization weight')
    group.add_argument(
        '--fedasync_staleness_func',
        type=str,
        default='constant',
        choices=['constant', 'poly', 'hinge'],
        help='FedAsync staleness weighting function',
    )
    group.add_argument('--fedasync_poly_a', type=float, default=0.5, help='FedAsync polynomial staleness exponent')
    group.add_argument('--fedasync_hinge_a', type=float, default=10.0, help='FedAsync hinge staleness slope')
    group.add_argument('--fedasync_hinge_b', type=float, default=4.0, help='FedAsync hinge staleness threshold')


def _add_adamasfl_args(parser):
    group = parser.add_argument_group('AdaMasFL')
    group.add_argument('--adamasfl_cache_device', choices=['device', 'cpu'], default=None,
                       help='legacy alias for --cache_device on AdaMasFL')
    group.add_argument('--adamasfl_beta', type=float, default=None, help='explicit AdaMasFL beta; otherwise fixed theorem formula')
    group.add_argument('--adamasfl_gamma', type=float, default=None, help='explicit AdaMasFL gamma; otherwise fixed theorem formula')
    group.add_argument('--adamasfl_lr', type=float, default=None, help='explicit AdaMasFL eta; otherwise fixed theorem formula')
    group.add_argument('--adamasfl_beta_scale', type=float, default=1.0, help='legacy theory-centre beta multiplier')
    group.add_argument('--adamasfl_gamma_scale', type=float, default=1.0, help='legacy theory-centre gamma multiplier')
    group.add_argument('--adamasfl_lr_scale', type=float, default=1.0, help='legacy theory-centre eta multiplier')




def args_parser():
    parser = argparse.ArgumentParser()

    _add_misc_args(parser)
    _add_federated_args(parser)
    _add_model_and_runtime_args(parser)
    _add_data_args(parser)
    _add_optimization_args(parser)
    _add_momentum_approximation_args(parser)
    _add_casa_args(parser)
    _add_fadas_args(parser)
    _add_fedasync_args(parser)
    _add_adamasfl_args(parser)

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.adamasfl_cache_device is not None:
        if args.algo != 'AdaMasFL':
            parser.error('--adamasfl_cache_device is only supported for AdaMasFL; use --cache_device')
        if (_flag_was_provided(argv, ['--cache_device', '--cache-device'])
                and args.cache_device != args.adamasfl_cache_device):
            parser.error('--cache_device conflicts with --adamasfl_cache_device')
        args.cache_device = args.adamasfl_cache_device
    args.adamasfl_cache_device = args.cache_device
    if args.saw_cache_topk <= 0 or args.saw_cache_power < 0 or args.saw_cache_exp_lambda < 0:
        parser.error('cache top-k must be positive; power and exponential rate must be nonnegative')
    return _apply_algo_defaults(args, argv)

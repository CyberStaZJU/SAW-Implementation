import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from algorithm import arrival_frozen_cache as frozen
from algorithm import saw


class ArrivalFrozenCacheTests(unittest.TestCase):
    def test_empty_cache(self):
        self.assertEqual(frozen._arrival_frozen_weights([None, None], []), [0.0, 0.0])

    def test_all_valid_cache_entries_are_normalized(self):
        weights = frozen._arrival_frozen_weights([2, 4, None, 2], [3])
        self.assertEqual(weights, [0.4, 0.2, 0.0, 0.4])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_new_entry_renormalizes_existing_weights(self):
        before = frozen._arrival_frozen_weights([2, 4, None], [])
        after = frozen._arrival_frozen_weights([2, 4, 1], [2])
        self.assertAlmostEqual(before[0], 2 / 3)
        self.assertAlmostEqual(after[0], 2 / 7)
        self.assertAlmostEqual(sum(after), 1.0)

    def make_state(self):
        return {
            'iterations': 8,
            'cost': [1, 2, -1],
            'stamp': [5, 7, -1],
            'arrival_age': [9, 2, None],
            'delays': [9, 1, None],
            'delta': [{'p': torch.tensor([1.0])}, {'p': torch.tensor([2.0])}, None],
            'cache': [{'p': torch.tensor([3.0])}, {'p': torch.tensor([4.0])}, None],
            'w_glob': {'p': torch.tensor([0.0])},
            'net_glob': object(),
        }

    def run_frozen(self, state):
        args = SimpleNamespace(num_users=3, buffer_size=1)
        with patch.object(frozen, '_local_train_delta', return_value={'p': torch.tensor([0.0])}), \
                patch.object(frozen, 'load_param_dict_'), \
                patch.object(frozen.np.random, 'shuffle', side_effect=lambda items: None):
            return frozen.run_round(state, args, lambda idx: 100)

    def test_cache_replacement_records_new_updates_age(self):
        state = self.run_frozen(self.make_state())
        self.assertEqual(state['arrival_age'][0], 4)
        self.assertEqual(state['cache'][0]['p'].item(), 1.0)
        self.assertEqual(state['iterations'], 9)

    def test_retained_update_keeps_age_across_rounds(self):
        state = self.run_frozen(self.make_state())
        self.assertEqual(state['arrival_age'][1], 2)
        state = self.run_frozen(state)
        self.assertEqual(state['arrival_age'][0], 4)
        self.assertEqual(state['arrival_age'][1], 3)
        self.assertAlmostEqual(sum(frozen._arrival_frozen_weights(state['arrival_age'], [])), 1.0)

    def test_saw_and_frozen_share_dispatch_events(self):
        frozen_state = self.make_state()
        saw_state = copy.deepcopy(frozen_state)
        self.run_frozen(frozen_state)
        args = SimpleNamespace(num_users=3, buffer_size=1, device='cpu', local_period=1)
        saw_state.update(dataset_train=[], dict_users=[[], [], []], num_samples=[0, 0, 0])
        net = unittest.mock.MagicMock()
        saw_state['net_glob'] = net
        with patch.object(saw, 'model_param_dict', return_value={'p': torch.tensor([0.0])}), \
                patch.object(saw, 'LocalSGD') as local, \
                patch.object(saw, 'load_param_dict_'), \
                patch.object(saw.np.random, 'shuffle', side_effect=lambda items: None):
            local.return_value.train.return_value = {'p': torch.tensor([0.0])}
            saw.run_round(saw_state, args, lambda idx: 100)
        for key in ('cost', 'stamp', 'iterations', 'global_cost'):
            self.assertEqual(frozen_state[key], saw_state[key])
        self.assertEqual(frozen_state['arrival_age'][0], saw_state['delays'][0])



class CacheDeviceTests(unittest.TestCase):
    METHODS = ('SAW', 'FedAsync', 'FedBuff', 'CA2FL', 'FedBuffMALight', 'CASA',
               'FADAS', 'AdaMasFL', 'SAW_EqualW', 'SAW_AdaInside', 'ArrivalFrozenCache',
               'SAW_UniformCache', 'SAW_FreshTopK', 'SAW_Power', 'SAW_Exp')

    def _parse(self, *flags):
        import sys
        from utils.options import args_parser
        with patch.object(sys, 'argv', ['test', *flags]):
            return args_parser()

    def test_cli_defaults_and_device_override(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                args = self._parse('--algo', method)
                self.assertEqual(args.cache_device, 'cpu')
                for flag in ('--cache_device', '--cache-device'):
                    args = self._parse('--algo', method, flag, 'device')
                    self.assertEqual(args.cache_device, 'device')
        for mode in ('cpu', 'device'):
            args = self._parse('--algo', 'AdaMasFL', '--adamasfl_cache_device', mode)
            self.assertEqual((args.cache_device, args.adamasfl_cache_device), (mode, mode))
        args = self._parse('--algo', 'AdaMasFL', '--adamasfl_cache_device=device')
        self.assertEqual((args.cache_device, args.adamasfl_cache_device), ('device', 'device'))
        args = self._parse('--algo', 'AdaMasFL', '--cache-device=cpu', '--adamasfl_cache_device=cpu')
        self.assertEqual((args.cache_device, args.adamasfl_cache_device), ('cpu', 'cpu'))

    def test_cli_rejects_conflicting_or_misapplied_alias(self):
        import contextlib
        import io
        for flags in (
            ('--algo', 'AdaMasFL', '--cache_device=cpu', '--adamasfl_cache_device=device'),
            ('--algo', 'AdaMasFL', '--adamasfl_cache_device=cpu', '--cache-device=device'),
            ('--algo', 'SAW', '--adamasfl_cache_device=cpu'),
            ('--cache_device', 'cuda'),
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self._parse(*flags)
                self.assertEqual(error.exception.code, 2)

    def test_legacy_cache_flag_is_rejected_for_every_non_ada_method(self):
        import contextlib
        import io
        for method in self.METHODS:
            if method == 'AdaMasFL':
                continue
            for mode in ('cpu', 'device'):
                with self.subTest(method=method, mode=mode), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self._parse('--algo', method, '--adamasfl_cache_device', mode)

    def test_cpu_snapshots_are_independent_and_detached(self):
        from utils.state_dict_ops import resolve_cache_device, sd_cache_zero_like, sd_to_cache
        values = {'p': torch.tensor([2.0], requires_grad=True)}
        cached = sd_to_cache(values, SimpleNamespace())
        self.assertFalse(cached['p'].requires_grad)
        self.assertNotEqual(cached['p'].data_ptr(), values['p'].data_ptr())
        with torch.no_grad():
            values['p'].fill_(3.0)
        self.assertEqual(cached['p'].item(), 2.0)
        self.assertIs(sd_to_cache(values, SimpleNamespace(cache_device='device')), values)
        self.assertEqual(sd_cache_zero_like(values, SimpleNamespace())['p'].item(), 0.0)
        self.assertEqual(resolve_cache_device(SimpleNamespace(adamasfl_cache_device='device')), 'device')
        with self.assertRaises(ValueError):
            resolve_cache_device(SimpleNamespace(cache_device='invalid'))

    def test_reductions_keep_add_then_scale_and_client_order(self):
        from utils.aggregation import buffered_aggregation
        from utils.state_dict_ops import sd_average, sd_axpy
        updates = [{'p': torch.tensor([value])} for value in (1e20, -1e20, 3.0)]
        ref = {'p': torch.zeros(1)}
        self.assertEqual(sd_average(updates, ref_model=ref)['p'].item(), 1.0)
        self.assertEqual(buffered_aggregation([1.0] * 3, updates, ref)['p'].item(), 3.0)
        result = {'p': torch.zeros(1)}
        for update in updates:
            sd_axpy(result, 1.0, update)
        self.assertEqual(result['p'].item(), 3.0)

    def test_fake_accelerator_cache_and_compute_placement(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from utils.aggregation import buffered_aggregation
        from utils.state_dict_ops import sd_average, sd_axpy, sd_cache_zero_like, sd_to_cache
        with FakeTensorMode():
            ref = {'p': torch.empty(2, device='cuda:0')}
            cached = sd_to_cache(ref, SimpleNamespace(cache_device='cpu'))
            zeros = sd_cache_zero_like(ref, SimpleNamespace(cache_device='cpu'))
            self.assertEqual(cached['p'].device.type, 'cpu')
            self.assertEqual(zeros['p'].device.type, 'cpu')
            self.assertEqual(sd_cache_zero_like(ref, SimpleNamespace(cache_device='device'))['p'].device, ref['p'].device)
            self.assertIs(sd_to_cache(ref, SimpleNamespace(cache_device='device')), ref)
            averaged = sd_average([cached, zeros], ref_model=ref)
            weighted = buffered_aggregation([0.2, 0.8], [cached, zeros], ref)
            sd_axpy(weighted, 0.3, cached)
            self.assertEqual(averaged['p'].device, ref['p'].device)
            self.assertEqual(weighted['p'].device, ref['p'].device)
            self.assertEqual(cached['p'].device.type, 'cpu')

    def test_all_methods_keep_fake_accelerator_server_state_and_cpu_clients(self):
        import contextlib
        from algorithm import dispatcher
        from torch._subclasses.fake_tensor import FakeTensorMode
        from utils.state_dict_ops import model_param_dict

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2, device='cuda:0'))
                self.register_buffer('running_mean', torch.empty(2, device='cuda:0'))
                self.register_buffer('num_batches_tracked', torch.zeros((), dtype=torch.long, device='cuda:0'))

            def __deepcopy__(self, memo):
                # Torch 2.8's FakeTensor deepcopy accesses an invalid data pointer.
                return type(self)()

        class FakeLocal:
            def __init__(self, args, **kwargs):
                self.args = args
                self.last_ec = {}

            def average_stochastic_gradients(self, net):
                return {name: torch.zeros_like(param) for name, param in net.named_parameters()
                        if param.requires_grad}

            def train(self, net):
                self.last_ec = self.average_stochastic_gradients(net)
                return model_param_dict(net, device=self.args.device)

        modules = (dispatcher.saw, dispatcher.fedasync, dispatcher.fedbuff, dispatcher.ca2fl,
                   dispatcher.fedbuffma, dispatcher.casa, dispatcher.fadas, dispatcher.saw_equalw,
                   dispatcher.saw_adainside, dispatcher.arrival_frozen_cache, dispatcher.saw_cache_ablation)
        for method in self.METHODS:
            for mode in ('cpu', 'device'):
                with self.subTest(method=method, mode=mode), FakeTensorMode(), contextlib.ExitStack() as stack:
                    for module in modules:
                        stack.enter_context(patch.object(module, 'LocalSGD', FakeLocal))
                        stack.enter_context(patch.object(module, 'load_param_dict_'))
                    stack.enter_context(patch.object(dispatcher.adamasfl, 'load_param_dict_'))
                    stack.enter_context(patch.object(dispatcher.adamasfl, '_CheckedLocalSGD', FakeLocal))
                    stack.enter_context(patch.object(dispatcher.adamasfl, '_CheckedLocalAdaMasFL', FakeLocal))
                    stack.enter_context(patch.object(dispatcher.adamasfl, '_validate_state'))
                    stack.enter_context(patch.object(dispatcher.adamasfl, '_validate_matching_state'))
                    # Placement only; numerical behavior is checked separately with real CPU tensors.
                    stack.enter_context(patch.object(dispatcher.casa, '_build_sketch_plan', return_value=[]))
                    stack.enter_context(patch.object(dispatcher.casa, '_maybe_recluster'))
                    stack.enter_context(patch.object(dispatcher.casa, '_encode_delta',
                                                    side_effect=lambda *args: torch.ones(2)))
                    stack.enter_context(patch.object(dispatcher.casa, '_feature_similarity', return_value=0.0))
                    stack.enter_context(patch.object(dispatcher, '_random_cost',
                                                    return_value=lambda idx=None: float(idx + 1)))
                    args = self._parse('--algo', method, '--cache_device', mode,
                                       '--num_users', '4', '--concurrency', '2', '--buffer_size', '1',
                                       '--local_period', '1', '--total_rounds', '8',
                                       '--adamasfl_beta', '0.3', '--random_cost', 'pareto')
                    args.device = 'cuda:0'
                    net = FakeModel()
                    state = dict(net_glob=net, w_glob=model_param_dict(net), dataset_train=None,
                                 dict_users={i: [] for i in range(4)}, num_samples=[2]*4, global_cost=0.0)
                    dispatcher.init_state(state, args)
                    for _ in range(3):
                        dispatcher.run_one_round(args, state, None, state['dict_users'], state['num_samples'])
                        for key in ('w_glob', 'adamasfl_c_global', 'adamasfl_g',
                                    'fadas_m', 'fadas_v', 'fadas_vhat'):
                            for value in state.get(key, {}).values():
                                self.assertEqual(value.device.type, 'cuda')
                        for key in ('delta', 'cache', 'local_models', 'adamasfl_c_local',
                                    'adamasfl_ec_local', 'adamasfl_buffer_local'):
                            for values in state.get(key, []):
                                if values is not None:
                                    for value in values.values():
                                        self.assertEqual(value.device.type, 'cpu' if mode == 'cpu' else 'cuda')
                        for value in state['net_glob'].state_dict().values():
                            self.assertEqual(value.device.type, 'cuda')
                        if 'ma_solver' in state:
                            for value in state['ma_solver'].prev_m.values():
                                self.assertEqual(value.device.type, 'cuda')

    def _small_run(self, method, mode):
        import random
        import numpy as np
        from algorithm import dispatcher
        from utils.state_dict_ops import model_param_dict
        args = self._parse('--algo', method, '--cache_device', mode,
                           '--num_users', '4', '--concurrency', '2', '--buffer_size', '1',
                           '--local_period', '1', '--local_bs', '2', '--total_rounds', '8',
                           '--lr', '0.02', '--global_lr', '0.3', '--momentum', '0.9',
                           '--adamasfl_beta', '0.3', '--adamasfl_gamma', '0.1', '--adamasfl_lr', '0.02',
                           '--ma_p', '0.5', '--ma_staleness_limit', '1',
                           '--random_cost', 'pareto')
        args.device = 'cpu'
        torch.manual_seed(43)
        random.seed(43)
        np.random.seed(43)
        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        dataset = torch.utils.data.TensorDataset(torch.randn(8, 2), torch.arange(8) % 2)
        users = {i: [2*i, 2*i+1] for i in range(4)}
        state = dict(net_glob=net, w_glob=model_param_dict(net), dataset_train=dataset,
                     dict_users=users, num_samples=[2]*4, global_cost=0.0)
        snapshots = []
        with patch.object(dispatcher, '_random_cost', return_value=lambda idx=None: float(idx + 1)):
            dispatcher.init_state(state, args)
            snapshots.append(self._snapshot(state))
            for _ in range(5):
                dispatcher.run_one_round(args, state, dataset, users, [2]*4)
                snapshots.append(self._snapshot(state))
        rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
        return state, snapshots, rng

    def _snapshot(self, state):
        keys = ('w_glob', 'delta', 'cache', 'local_models', 'adamasfl_c_local',
                'adamasfl_ec_local', 'adamasfl_buffer_local', 'adamasfl_c_global',
                'adamasfl_g', 'fadas_m', 'fadas_v', 'fadas_vhat', 'client_features',
                'client_buffers', 'similarity', 'group_ids', 'group_members',
                'cost', 'stamp', 'delays', 'arrival_age', 'iterations', 'global_cost')
        snapshot = {key: copy.deepcopy(state[key]) for key in keys if key in state}
        snapshot['model'] = copy.deepcopy(state['net_glob'].state_dict())
        if 'ma_solver' in state:
            solver = state['ma_solver']
            snapshot['ma'] = copy.deepcopy((solver.W, solver.light_A, solver.prev_a, solver.prev_m,
                                            solver.last_u, solver.last_v))
        return snapshot

    def _assert_equal(self, actual, expected):
        import numpy as np
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self._assert_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self._assert_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_all_retained_methods_preserve_seeded_cpu_recurrence_bn_and_rng(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                _, device_steps, device_rng = self._small_run(method, 'device')
                state, cpu_steps, cpu_rng = self._small_run(method, 'cpu')
                self._assert_equal(cpu_steps, device_steps)
                self._assert_equal(cpu_rng, device_rng)
                self.assertEqual(state['cache_device'], 'cpu')
                for key in ('delta', 'cache', 'local_models', 'adamasfl_c_local',
                            'adamasfl_ec_local', 'adamasfl_buffer_local'):
                    for values in state.get(key, []):
                        if values is not None:
                            self.assertTrue(all(v.device.type == 'cpu' for v in values.values()))

    def test_removed_methods_are_not_dispatched(self):
        from algorithm import dispatcher
        removed = ('FedAC', 'DirBridge', 'FedASMU', 'OrMo', 'FedBuffMA',
                   'FedStaleWeight', 'AsyncFedED', 'WKAFL', 'FedCompass',
                   'FedFa', 'ACE', 'ACED', 'FedSteer')
        for method in removed:
            args = self._parse('--algo', method, '--random_cost', 'pareto')
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'Unsupported algorithm'):
                    dispatcher.init_state({}, args)
                with self.assertRaisesRegex(ValueError, 'Unsupported algorithm'):
                    dispatcher.run_one_round(args, {}, None, None, None)
        from algorithm.fedbuffma import MomentumApproximation
        self.assertFalse(hasattr(MomentumApproximation, 'solve_full'))


class CacheAblationWeightsTests(unittest.TestCase):
    def test_topk_breaks_equal_age_ties_by_client_id(self):
        from algorithm.saw_cache_ablation import _select_topk_fresh, _cache_weights
        delays = [4, 1, 1, None, 1, 0, 1]
        self.assertEqual(_select_topk_fresh(delays, [6, 5], 3), [5, 1, 2])
        args = SimpleNamespace(algo='SAW_FreshTopK', saw_cache_topk=3)
        self.assertEqual(_cache_weights(delays, [6, 5], args), [0, 1/3, 1/3, 0, 0, 1/3, 0])
        defaults = SimpleNamespace(algo='SAW_FreshTopK')
        self.assertEqual(sum(value > 0 for value in _cache_weights(delays, [6, 5], defaults)), 5)

    def test_power_zero_one_half_and_two_match_age_formula(self):
        from algorithm.saw_cache_ablation import _cache_weights
        from utils.aggregation import rescaling
        delays = [0, 3, None, 8]
        for power in (0.0, 0.5, 1.0, 2.0):
            args = SimpleNamespace(algo='SAW_Power', saw_cache_power=power)
            weights = _cache_weights(delays, [0], args)
            raw = [1.0, 4.0 ** (-power), 9.0 ** (-power)]
            expected = [raw[0]/sum(raw), raw[1]/sum(raw), 0.0, raw[2]/sum(raw)]
            self.assertEqual(weights, expected)
            if power == 1.0:
                self.assertEqual(weights, rescaling(delays, [0]))

    def test_exp_uses_one_based_age_and_point_two_default(self):
        import math
        from algorithm.saw_cache_ablation import _cache_weights
        delays = [0, None, 2, 9]
        for lam in (0.0, 0.2):
            args = SimpleNamespace(algo='SAW_Exp', saw_cache_exp_lambda=lam)
            raw = [math.exp(-lam * age) for age in (1, 3, 10)]
            expected = [raw[0]/sum(raw), 0.0, raw[1]/sum(raw), raw[2]/sum(raw)]
            self.assertEqual(_cache_weights(delays, [0], args), expected)
        self.assertEqual(_cache_weights(delays, [0], SimpleNamespace(algo='SAW_Exp')),
                         _cache_weights(delays, [0], SimpleNamespace(algo='SAW_Exp', saw_cache_exp_lambda=0.2)))

    def test_uniform_and_empty_cache_mask_invalid_entries(self):
        from algorithm.saw_cache_ablation import SAW_CACHE_ABLATION_ALGOS, _cache_weights
        for method in SAW_CACHE_ABLATION_ALGOS:
            self.assertEqual(_cache_weights([None, None], [], SimpleNamespace(algo=method)), [0.0, 0.0])
        self.assertEqual(_cache_weights([0, None, 5], [0], SimpleNamespace(algo='SAW_UniformCache')),
                         [0.5, 0.0, 0.5])

if __name__ == '__main__':
    unittest.main()

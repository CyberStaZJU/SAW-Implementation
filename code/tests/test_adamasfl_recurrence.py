import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from algorithm import adamasfl
from models.local_update import LocalAdaMasFL
from utils.state_dict_ops import model_param_dict, sd_copy


class _BNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2, momentum=1.0)
        self.head = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.register_buffer('feature_mean', torch.zeros(2))
        self.register_buffer('sample_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('seen', torch.tensor(False))
        self.register_buffer('scratch', torch.tensor(0), persistent=False)

    def forward(self, images):
        if self.training:
            self.feature_mean.copy_(images.detach().mean(0))
            self.sample_count.add_(images.shape[0])
            self.seen.fill_(True)
            self.scratch.add_(1)
        return self.head(self.bn(images)) + self.frozen


class AdamasflRecurrenceTests(unittest.TestCase):
    def _net(self):
        return torch.nn.Linear(2, 1, bias=False)

    def _state(self, cls, beta=0.3):
        obj = cls.__new__(cls)
        obj.momentum_beta = beta
        obj.local_c = {'weight': torch.tensor([[1.0, 2.0]])}
        obj.server_c = {'weight': torch.tensor([[0.5, 1.5]])}
        obj.g_mix = {'weight': torch.tensor([[0.25, 0.75]])}
        obj.last_ec = {}
        obj.sum_grads = {'weight': torch.zeros_like(obj.local_c['weight'])}
        return obj

    def test_adamasfl_normalizes_the_full_direction(self):
        net = self._net()
        net.weight.grad = torch.tensor([[4.0, 8.0]])
        obj = self._state(LocalAdaMasFL)
        obj.g_next = {}
        obj.sum_grads = {'weight': torch.zeros_like(net.weight)}
        obj.modify_gradients(net)
        direction = 0.3 * (torch.tensor([[4.0, 8.0]]) - torch.tensor([[1.0, 2.0]]) + torch.tensor([[0.5, 1.5]])) + 0.7 * torch.tensor([[0.25, 0.75]])
        torch.testing.assert_close(net.weight.grad, direction / torch.linalg.vector_norm(direction))


    def test_resolved_arguments_preserve_delay_config_without_runtime_objects(self):
        import json
        from algorithm.dispatcher import _ensure_client_delay_profile
        from main_fed import resolved_arguments
        args = SimpleNamespace(num_users=4, seed=7, delay_seed=11, delay_gamma=1.0,
                               random_cost='hierarchical', device=torch.device('cpu'))
        _ensure_client_delay_profile(args)
        args.fedscale_trace_sampler = object()
        rng = args.client_delay_rng
        payload = json.loads(json.dumps(resolved_arguments(args)))
        self.assertEqual(payload['device'], 'cpu')
        self.assertEqual(payload['delay_seed'], 11)
        self.assertEqual(payload['client_delay_groups'], args.client_delay_groups)
        self.assertEqual(payload['client_delay_group_probs'], args.client_delay_group_probs)
        self.assertNotIn('client_delay_rng', payload)
        self.assertNotIn('fedscale_trace_sampler', payload)
        self.assertIs(args.client_delay_rng, rng)

    def test_evaluation_rejects_nonfinite_accuracy_or_loss(self):
        from main_fed import evaluate_global_model
        state = {'net_glob': torch.nn.Linear(2, 2)}
        args = SimpleNamespace(algo='AdaMasFL', dataset='cifar')
        for result in ((10.0, float('nan')), (float('nan'), 1.0), (10.0, float('inf'))):
            with self.subTest(result=result), patch('main_fed.test_img', return_value=result):
                with self.assertRaisesRegex(ValueError, 'evaluation.*non-finite'):
                    evaluate_global_model(state, None, args)
        with patch('main_fed.test_img', return_value=(50.0, 1.0)):
            self.assertEqual(evaluate_global_model(state, None, args), (50.0, 1.0))

    def test_cli_and_dispatch_complete_one_round(self):
        import sys
        from unittest.mock import patch
        from algorithm import dispatcher
        from utils.options import args_parser

        argv = ['test', '--algo', 'AdaMasFL', '--num_users', '4',
                '--concurrency', '2', '--buffer_size', '1', '--local_period', '1',
                '--local_bs', '2', '--num_workers', '0', '--total_rounds', '500',
                '--adamasfl_beta', '0.3', '--adamasfl_lr', '0.008',
                '--adamasfl_gamma', '0.1']
        with patch.object(sys, 'argv', argv):
            args = args_parser()
        args.device = 'cpu'
        torch.manual_seed(3)
        dataset = torch.utils.data.TensorDataset(torch.randn(8, 2), torch.arange(8) % 2)
        users = {i: [2*i, 2*i+1] for i in range(4)}
        net = torch.nn.Linear(2, 2)
        initial = net.weight.detach().clone()
        state = {'net_glob': net, 'w_glob': {k: v.clone() for k, v in net.state_dict().items()},
                 'dataset_train': dataset, 'dict_users': users,
                 'num_samples': [2]*4, 'global_cost': 0.0}
        with patch.object(dispatcher, '_random_cost', return_value=lambda idx=None: 1.0):
            dispatcher.init_state(state, args)
            dispatcher.run_one_round(args, state, dataset, users, [2]*4)
        self.assertEqual(state['iterations'], 1)
        self.assertEqual(state['adamasfl_hparams'], {'beta': 0.3, 'gamma': 0.1, 'eta': 0.008})
        self.assertFalse(torch.equal(initial, net.weight))
        self.assertTrue(all(torch.isfinite(v).all() for v in state['w_glob'].values()))


class AdamasflStateSeparationTests(unittest.TestCase):
    def _args(self, **overrides):
        values = dict(device='cpu', num_users=3, concurrency=0, buffer_size=2,
                      local_period=1, local_bs=2, num_workers=0, local_epochs=None,
                      total_rounds=500, lr=0.008, weight_decay=0.0,
                      adamasfl_beta=0.3, adamasfl_lr=0.008, adamasfl_gamma=0.1)
        values.update(overrides)
        return SimpleNamespace(**values)

    def _state(self, net=None):
        torch.manual_seed(19)
        net = _BNModel() if net is None else net
        images = torch.tensor([[0.0, 0.0], [0.2, 0.4],
                               [1.0, 2.0], [1.4, 2.6],
                               [-10.0, -20.0], [10.0, 20.0]])
        dataset = torch.utils.data.TensorDataset(images, torch.tensor([0, 1, 0, 1, 0, 1]))
        return dict(net_glob=net, w_glob=model_param_dict(net), dataset_train=dataset,
                    dict_users={i: [2*i, 2*i+1] for i in range(3)},
                    num_samples=[2, 2, 2], global_cost=0.0)

    def _run(self, state, args):
        return adamasfl.run_round(state, args, state['dataset_train'],
                                  state['dict_users'], state['num_samples'], lambda idx: 1.0)

    def _assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def _set_global_buffers(self, state, mean, variance, count, sample_count):
        state['w_glob']['bn.running_mean'].fill_(mean)
        state['w_glob']['bn.running_var'].fill_(variance)
        state['w_glob']['bn.num_batches_tracked'].fill_(count)
        state['w_glob']['sample_count'].fill_(sample_count)
        state['net_glob'].load_state_dict(state['w_glob'], strict=True)

    def _stale_state(self):
        args = self._args()
        state = self._state()
        adamasfl.init_state(state, args, lambda idx: 1.0)
        starts, trained = [], []
        train = adamasfl._CheckedLocalAdaMasFL.train

        def capture(local, net):
            result = train(local, net)
            trained.append(sd_copy(net.state_dict()))
            return result

        with patch.object(adamasfl._CheckedLocalAdaMasFL, 'train', new=capture):
            starts.append(sd_copy(state['net_glob'].state_dict()))
            adamasfl._launch_adamasfl_job(state, args, 0, 0.0, lambda idx: 1.0)
            self._set_global_buffers(state, 7.0, 4.0, 20, 40)
            starts.append(sd_copy(state['net_glob'].state_dict()))
            adamasfl._launch_adamasfl_job(state, args, 1, 0.0, lambda idx: 2.0)
            self._set_global_buffers(state, 9.0, 7.0, 1000, 2000)
            starts.append(sd_copy(state['net_glob'].state_dict()))
            adamasfl._launch_adamasfl_job(state, args, 2, 0.0, lambda idx: 100.0)
        self._set_global_buffers(state, 9.0, 1.0, 50, 100)
        return state, args, starts, trained

    def test_initial_control_keeps_global_bn_and_exports_full_state(self):
        for trainable_only in (False, True):
            with self.subTest(trainable_only=trainable_only):
                state = self._state()
                net = state['net_glob']
                net._asyncbuffer_trainable_state_only = trainable_only
                state['w_glob'] = model_param_dict(net)
                original = sd_copy(net.state_dict())
                scratch = net.scratch.clone()
                adamasfl.init_state(state, self._args(), lambda idx: 1.0)
                self._assert_state_equal(net.state_dict(), original)
                self._assert_state_equal(state['w_glob'], original)
                torch.testing.assert_close(net.scratch, scratch)
                params = {name for name, value in net.named_parameters() if value.requires_grad}
                self.assertEqual(set(state['adamasfl_param_names']), params)
                self.assertIn('bn.weight', params)
                self.assertIn('bn.bias', params)
                self.assertNotIn('frozen', params)
                for key in ('delta', 'adamasfl_c_local', 'adamasfl_ec_local'):
                    for vector in state[key]:
                        self.assertEqual(set(vector), params)
                for key in ('adamasfl_c_global', 'adamasfl_g'):
                    self.assertEqual(set(state[key]), params)
                consumer = _BNModel()
                consumer.load_state_dict(state['w_glob'], strict=True)
                consumer.eval()
                self.assertTrue(torch.isfinite(consumer(torch.randn(2, 2))).all())
                self.assertEqual(net._asyncbuffer_trainable_state_only, trainable_only)
                if not trainable_only:
                    self.assertIn('bn.running_var', model_param_dict(net))

    def test_init_launches_real_local_jobs_without_polluting_global_bn(self):
        state = self._state()
        original = sd_copy(state['net_glob'].state_dict())
        args = self._args(concurrency=2)
        with patch.object(adamasfl.random, 'sample', return_value=[0, 1]):
            adamasfl.init_state(state, args, lambda idx: idx + 1.0)
        self._assert_state_equal(state['net_glob'].state_dict(), original)
        self.assertEqual(state['cost'], [1.0, 2.0, -1])
        torch.testing.assert_close(state['adamasfl_buffer_local'][0]['bn.running_var'],
                                   torch.tensor([0.02, 0.08]))
        torch.testing.assert_close(state['adamasfl_buffer_local'][1]['bn.running_var'],
                                   torch.tensor([0.08, 0.18]))
        self.assertEqual(state['adamasfl_buffer_local'][0]['bn.num_batches_tracked'].item(), 1)
        self.assertFalse(getattr(state['net_glob'], '_asyncbuffer_trainable_state_only', False))

    def test_two_stale_bn_jobs_use_absolute_stats_and_parameter_only_update(self):
        state, args, starts, trained = self._stale_state()
        params = state['adamasfl_param_names']
        for idx in range(3):
            self._assert_state_equal(state['adamasfl_buffer_local'][idx],
                                     {name: value for name, value in trained[idx].items()
                                      if name in dict(state['net_glob'].named_buffers())})
            for name in params:
                torch.testing.assert_close(state['delta'][idx][name],
                                           trained[idx][name] - starts[idx][name])
        factor = state['gamma'] / (args.lr * args.local_period)
        self.assertGreater(factor, 1.0)
        old_variance = state['w_glob']['bn.running_var'] + factor * (
            (trained[0]['bn.running_var'] - starts[0]['bn.running_var']) +
            (trained[1]['bn.running_var'] - starts[1]['bn.running_var'])) / 2
        self.assertTrue((old_variance < 0).all())

        before = sd_copy(state['w_glob'])
        state['num_samples'] = [2, 6, 10]
        c_before = sd_copy(state['adamasfl_c_global'])
        g_before = sd_copy(state['adamasfl_g'])
        local_before = copy.deepcopy(state['adamasfl_c_local'])
        expected = {name: before[name] + factor * (
            state['delta'][0][name] + state['delta'][1][name]) / 2 for name in params}
        expected_c = {}
        expected_g = {}
        for name in params:
            diff = (state['adamasfl_ec_local'][0][name] - local_before[0][name] +
                    state['adamasfl_ec_local'][1][name] - local_before[1][name])
            expected_c[name] = c_before[name] + diff / args.num_users
            expected_g[name] = state['beta'] * (c_before[name] + diff / 2) + (
                1 - state['beta']) * g_before[name]

        with patch.object(adamasfl, '_launch_adamasfl_job') as launch:
            self._run(state, args)
        launch.assert_not_called()
        self.assertEqual(state['last_buffer_list'], [0, 1])
        self.assertEqual(state['iterations'], 1)
        self.assertEqual(state['global_cost'], 2.0)
        self.assertEqual(state['cost'], [-1, -1, 100.0])
        for name in params:
            torch.testing.assert_close(state['w_glob'][name], expected[name])
            torch.testing.assert_close(state['adamasfl_c_global'][name], expected_c[name])
            torch.testing.assert_close(state['adamasfl_g'][name], expected_g[name])
        for name in ('bn.running_mean', 'bn.running_var', 'feature_mean'):
            torch.testing.assert_close(state['w_glob'][name],
                                       (trained[0][name] + trained[1][name]) / 2)
        torch.testing.assert_close(state['w_glob']['bn.running_var'], torch.tensor([0.05, 0.13]))
        self.assertTrue((state['w_glob']['bn.running_var'] >= 0).all())
        self.assertEqual(state['w_glob']['bn.num_batches_tracked'].item(), 21)
        self.assertEqual(state['w_glob']['bn.num_batches_tracked'].dtype, torch.long)
        self.assertEqual(state['w_glob']['sample_count'].item(), 42)
        self.assertEqual(state['w_glob']['sample_count'].dtype, torch.long)
        self.assertTrue(state['w_glob']['seen'].item())
        self.assertEqual(state['w_glob']['seen'].dtype, torch.bool)
        torch.testing.assert_close(state['w_glob']['frozen'], before['frozen'], rtol=0, atol=0)
        self._assert_state_equal(state['adamasfl_c_local'][2], local_before[2])
        self._assert_state_equal(state['net_glob'].state_dict(), state['w_glob'])
        self.assertEqual(state['net_glob'].scratch.item(), 0)
        consumer = _BNModel()
        consumer.load_state_dict(state['w_glob'], strict=True)
        consumer.eval()
        state['net_glob'].eval()
        images = state['dataset_train'].tensors[0]
        torch.testing.assert_close(consumer(images), state['net_glob'](images))
        self.assertTrue(torch.isfinite(consumer(images)).all())

    def test_linear_local_path_matches_shared_recurrence(self):
        state = self._state(torch.nn.Linear(2, 2))
        args = self._args(local_period=2)
        adamasfl.init_state(state, args, lambda idx: 1.0)
        reference = copy.deepcopy(state['net_glob'])
        local = LocalAdaMasFL(args=args, dataset=state['dataset_train'],
                             idxs=state['dict_users'][0], iters=args.local_period, nums=2,
                             momentum_beta=state['beta'], local_c=state['adamasfl_c_local'][0],
                             server_c=state['adamasfl_c_global'], g_mix=state['adamasfl_g'])
        rng = torch.get_rng_state()
        initial = sd_copy(reference.state_dict())
        expected = local.train(reference)
        torch.set_rng_state(rng)
        adamasfl._launch_adamasfl_job(state, args, 0, 0.0, lambda idx: 1.0)
        self.assertEqual(state['adamasfl_buffer_local'][0], {})
        for name in initial:
            torch.testing.assert_close(state['delta'][0][name], expected[name] - initial[name],
                                       rtol=0, atol=0)
            torch.testing.assert_close(state['adamasfl_ec_local'][0][name], local.last_ec[name],
                                       rtol=0, atol=0)
        before = sd_copy(state['w_glob'])
        with patch.object(adamasfl, '_launch_adamasfl_job'):
            self._run(state, args)
        factor = state['gamma'] / (args.lr * args.local_period)
        for name in initial:
            torch.testing.assert_close(state['w_glob'][name],
                                       before[name] + factor * (expected[name] - initial[name]))
        self.assertEqual(state['last_buffer_list'], [0])

    def test_integer_buffer_maxima_preserve_large_counts_exactly(self):
        net = _BNModel()
        snapshots = [adamasfl._buffer_state(net) for _ in range(2)]
        for idx, snapshot in enumerate(snapshots):
            snapshot['bn.num_batches_tracked'].fill_(2**60 + idx)
            snapshot['sample_count'].fill_(2**60 + 3 - idx)
            snapshot['seen'].fill_(bool(idx))
        averaged = adamasfl._average_buffers(net, snapshots)
        self.assertEqual(averaged['bn.num_batches_tracked'].item(), 2**60 + 1)
        self.assertEqual(averaged['sample_count'].item(), 2**60 + 3)
        self.assertEqual(averaged['sample_count'].dtype, torch.long)
        self.assertTrue(averaged['seen'].item())
        self.assertEqual(averaged['seen'].dtype, torch.bool)

    def test_bn_average_is_independent_of_server_gamma(self):
        state, args, _, _ = self._stale_state()
        other = copy.deepcopy(state)
        other['gamma'] *= 2
        with patch.object(adamasfl, '_launch_adamasfl_job'):
            self._run(state, args)
            self._run(other, args)
        for name in adamasfl._buffer_state(state['net_glob']):
            torch.testing.assert_close(state['w_glob'][name], other['w_glob'][name], rtol=0, atol=0)
        self.assertFalse(torch.equal(state['w_glob']['head.weight'], other['w_glob']['head.weight']))

    def test_control_recurrence_keeps_buffer_and_population_denominators(self):
        state = self._state(torch.nn.Linear(2, 2))
        args = self._args()
        adamasfl.init_state(state, args, lambda idx: 1.0)
        state['cost'] = [1.0, 2.0, 100.0]
        for name in state['adamasfl_param_names']:
            state['adamasfl_c_global'][name].fill_(10)
            state['adamasfl_g'][name].fill_(20)
            for idx, (old, new) in enumerate(((1, 4), (2, 8), (7, 100))):
                state['adamasfl_c_local'][idx][name].fill_(old)
                state['adamasfl_ec_local'][idx][name].fill_(new)
        with patch.object(adamasfl, '_launch_adamasfl_job'):
            self._run(state, args)
        for name in state['adamasfl_param_names']:
            torch.testing.assert_close(state['adamasfl_c_global'][name],
                                       torch.full_like(state['adamasfl_c_global'][name], 13))
            torch.testing.assert_close(state['adamasfl_g'][name],
                                       torch.full_like(state['adamasfl_g'][name], 18.35))
            self.assertTrue((state['adamasfl_c_local'][0][name] == 4).all())
            self.assertTrue((state['adamasfl_c_local'][1][name] == 8).all())
            self.assertTrue((state['adamasfl_c_local'][2][name] == 7).all())

    def test_multilayer_direction_uses_one_global_norm_and_stale_recurrence(self):
        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1, bias=False))
        net[0].bias.requires_grad_(False)
        params = {name: value for name, value in net.named_parameters() if value.requires_grad}
        local = adamasfl._CheckedLocalAdaMasFL.__new__(adamasfl._CheckedLocalAdaMasFL)
        local.momentum_beta = 0.3
        local.local_c = {name: torch.full_like(value, 1) for name, value in params.items()}
        local.server_c = {name: torch.full_like(value, 2) for name, value in params.items()}
        local.g_mix = {name: torch.full_like(value, 3) for name, value in params.items()}
        local.g_next = {}
        local.sum_grads = {name: torch.zeros_like(value) for name, value in params.items()}
        total_grads = {name: torch.zeros_like(value) for name, value in params.items()}
        for scale in (1.0, 2.0):
            raw = {'0.weight': scale * torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
                   '1.weight': scale * torch.tensor([[0.0, 12.0]])}
            direction = {name: 0.3 * (grad - 1 + 2) + 0.7 * 3 for name, grad in raw.items()}
            flat = torch.cat([value.flatten() for value in direction.values()])
            for name, value in params.items():
                value.grad = raw[name].clone()
                total_grads[name] += raw[name]
            local.modify_gradients(net)
            for name, value in params.items():
                torch.testing.assert_close(value.grad, direction[name] / torch.linalg.vector_norm(flat))
                torch.testing.assert_close(local.g_next[name], direction[name])
                torch.testing.assert_close(local.sum_grads[name], total_grads[name])
            self.assertIsNone(net[0].bias.grad)
        local.last_ec = {}
        local.after_train(net, 2)
        for name in params:
            torch.testing.assert_close(local.last_ec[name], total_grads[name] / 2)

    def test_nonfinite_local_gradient_and_direction_are_rejected(self):
        for mode in ('gradient', 'direction', 'norm'):
            with self.subTest(mode=mode):
                net = torch.nn.Linear(2, 1, bias=False)
                local = AdamasflRecurrenceTests()._state(adamasfl._CheckedLocalAdaMasFL)
                local.g_next = {}
                if mode == 'gradient':
                    net.weight.grad = torch.tensor([[float('nan'), 1.0]])
                else:
                    net.weight.grad = torch.ones_like(net.weight)
                    if mode == 'direction':
                        local.momentum_beta = 0.9
                        local.local_c['weight'].fill_(-3e38)
                        net.weight.grad.fill_(3e38)
                    else:
                        net.weight.grad.fill_(1e30)
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    local.modify_gradients(net)

    def test_nonfinite_initial_estimation_fails_without_changing_global_bn(self):
        for mode in ('device', 'cpu'):
            with self.subTest(mode=mode):
                state = self._state()
                original = sd_copy(state['net_glob'].state_dict())
                state['dataset_train'].tensors[0][0, 0] = float('nan')
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    adamasfl.init_state(state, self._args(adamasfl_cache_device=mode), lambda idx: 1.0)
                self._assert_state_equal(state['net_glob'].state_dict(), original)

    def test_launch_rejects_nonfinite_loss_without_enqueuing_a_job(self):
        state = self._state()
        args = self._args()
        adamasfl.init_state(state, args, lambda idx: 1.0)
        before = sd_copy(state['net_glob'].state_dict())
        state['dataset_train'].tensors[0][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            adamasfl._launch_adamasfl_job(state, args, 0, 0.0, lambda idx: 1.0)
        self.assertEqual(state['cost'][0], -1)
        self.assertEqual(state['adamasfl_buffer_local'][0], {})
        self._assert_state_equal(state['net_glob'].state_dict(), before)

    def test_launch_validates_actual_local_parameters_and_buffers(self):
        state = self._state()
        args = self._args()
        adamasfl.init_state(state, args, lambda idx: 1.0)
        original = sd_copy(state['net_glob'].state_dict())
        train = adamasfl._CheckedLocalAdaMasFL.train
        for name, value in (('head.weight', float('nan')),
                            ('bn.running_mean', float('inf')),
                            ('bn.running_var', -0.1)):
            with self.subTest(name=name):
                def corrupt(local, net):
                    result = train(local, net)
                    with torch.no_grad():
                        net.state_dict()[name].fill_(value)
                    if name in result:
                        result[name].fill_(value)
                    return result
                with patch.object(adamasfl._CheckedLocalAdaMasFL, 'train', new=corrupt):
                    with self.assertRaisesRegex(ValueError, 'non-finite|negative'):
                        adamasfl._launch_adamasfl_job(state, args, 0, 0.0, lambda idx: 1.0)
                self.assertEqual(state['cost'][0], -1)
                self._assert_state_equal(state['net_glob'].state_dict(), original)

    def test_round_rejects_invalid_selected_state_before_committing(self):
        base, args, _, _ = self._stale_state()
        cases = [('buffer', 'bn.running_var', -0.01),
                 ('buffer', 'bn.running_var', float('nan')),
                 ('buffer', 'bn.running_mean', float('inf')),
                 ('buffer', 'bn.num_batches_tracked', -1),
                 ('buffer', 'feature_mean', float('nan')),
                 ('delta', 'head.weight', float('nan')),
                 ('control', 'bn.weight', float('inf'))]
        for kind, name, value in cases:
            with self.subTest(kind=kind, name=name, value=value):
                state = copy.deepcopy(base)
                key = {'buffer': 'adamasfl_buffer_local', 'delta': 'delta',
                       'control': 'adamasfl_ec_local'}[kind]
                state[key][0][name].fill_(value)
                before = sd_copy(state['w_glob'])
                costs = list(state['cost'])
                old_c = sd_copy(state['adamasfl_c_global'])
                with patch.object(adamasfl, '_launch_adamasfl_job') as launch:
                    with self.assertRaisesRegex(ValueError, 'non-finite|negative'):
                        self._run(state, args)
                launch.assert_not_called()
                self.assertEqual(state['iterations'], 0)
                self.assertEqual(state['cost'], costs)
                self._assert_state_equal(state['w_glob'], before)
                self._assert_state_equal(state['net_glob'].state_dict(), before)
                self._assert_state_equal(state['adamasfl_c_global'], old_c)

    def test_finite_delta_that_overflows_server_direction_is_rejected(self):
        state, args, _, _ = self._stale_state()
        state['delta'][0]['head.weight'].fill_(3e38)
        before = sd_copy(state['w_glob'])
        with patch.object(adamasfl, '_launch_adamasfl_job'):
            with self.assertRaisesRegex(ValueError, 'server direction.*non-finite'):
                self._run(state, args)
        self._assert_state_equal(state['w_glob'], before)
        self.assertEqual(state['iterations'], 0)


class AdamasflCacheResidencyTests(unittest.TestCase):
    def test_initial_estimation_keeps_client_order_and_model_state(self):
        helpers = AdamasflStateSeparationTests()
        estimate = adamasfl._estimate_initial_control
        for mode in ('cpu', 'device'):
            with self.subTest(mode=mode):
                state = helpers._state()
                original = sd_copy(state['net_glob'].state_dict())
                scratch = state['net_glob'].scratch.clone()
                args = helpers._args(adamasfl_cache_device=mode, lr=0.5)
                clients = []

                def checked_estimate(current, actual_args, idx):
                    clients.append(idx)
                    self.assertEqual(actual_args.lr, args.adamasfl_lr)
                    if idx == 0:
                        for key in ('delta', 'adamasfl_c_local', 'adamasfl_ec_local'):
                            if mode == 'cpu':
                                self.assertNotIn(key, current)
                            else:
                                self.assertEqual(len(current[key]), args.num_users)
                    return estimate(current, actual_args, idx)

                with patch.object(adamasfl, '_estimate_initial_control', side_effect=checked_estimate), \
                        patch.object(adamasfl, '_resolve_adamasfl_hparams', wraps=adamasfl._resolve_adamasfl_hparams) as resolve:
                    adamasfl.init_state(state, args, lambda idx: 1.0)
                resolve.assert_called_once()
                self.assertEqual(clients, list(range(args.num_users)))
                helpers._assert_state_equal(state['net_glob'].state_dict(), original)
                torch.testing.assert_close(state['net_glob'].scratch, scratch, rtol=0, atol=0)
                for control, returned in zip(state['adamasfl_c_local'], state['adamasfl_ec_local']):
                    helpers._assert_state_equal(control, returned)
                    for name in control:
                        self.assertNotEqual(control[name].data_ptr(), returned[name].data_ptr())

    def test_cpu_cache_has_independent_storage(self):
        values = {'weight': torch.tensor([1.0, 2.0])}
        cached = adamasfl._cache_state({'adamasfl_cache_device': 'cpu'}, values)
        self.assertNotEqual(cached['weight'].data_ptr(), values['weight'].data_ptr())
        values['weight'].fill_(9)
        torch.testing.assert_close(cached['weight'], torch.tensor([1.0, 2.0]), rtol=0, atol=0)
        self.assertIs(adamasfl._cache_state({}, values), values)

    def test_cpu_cache_preserves_seeded_multiround_recurrence(self):
        import random
        import numpy as np
        helpers = AdamasflStateSeparationTests()
        initial = helpers._state()
        outcomes = []
        rng_states = []
        for mode in ('device', 'cpu'):
            torch.manual_seed(31)
            random.seed(31)
            np.random.seed(31)
            state = copy.deepcopy(initial)
            args = helpers._args(concurrency=2, buffer_size=1, adamasfl_cache_device=mode)
            adamasfl.init_state(state, args, lambda idx: float(idx + 1))
            initialized_rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
            for _ in range(5):
                adamasfl.run_round(state, args, state['dataset_train'], state['dict_users'],
                                  state['num_samples'], lambda idx: float(idx + 1))
            rng_states.append((initialized_rng, (torch.get_rng_state(), random.getstate(), np.random.get_state())))
            outcomes.append(state)
        for actual, expected in zip(rng_states[0], rng_states[1]):
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertEqual(actual[1], expected[1])
            self.assertEqual(actual[2][0], expected[2][0])
            np.testing.assert_array_equal(actual[2][1], expected[2][1])
            self.assertEqual(actual[2][2:], expected[2][2:])
        left, right = outcomes
        for key in ('w_glob', 'adamasfl_c_global', 'adamasfl_g'):
            helpers._assert_state_equal(left[key], right[key])
        for key in ('delta', 'adamasfl_c_local', 'adamasfl_ec_local', 'adamasfl_buffer_local'):
            for actual, expected in zip(left[key], right[key]):
                helpers._assert_state_equal(actual, expected)
        for key in ('cost', 'global_cost', 'iterations', 'last_buffer_list'):
            self.assertEqual(left[key], right[key])
        for control in right['adamasfl_c_local']:
            self.assertTrue(all(value.device.type == 'cpu' for value in control.values()))

    def test_cpu_cache_rejects_corrupted_snapshots_before_commit(self):
        helpers = AdamasflStateSeparationTests()
        state, args, _, _ = helpers._stale_state()
        state['adamasfl_cache_device'] = 'cpu'
        state['delta'][0]['head.weight'].fill_(float('nan'))
        original = sd_copy(state['w_glob'])
        with patch.object(adamasfl, '_launch_adamasfl_job') as launch:
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                helpers._run(state, args)
        launch.assert_not_called()
        helpers._assert_state_equal(state['w_glob'], original)
        self.assertEqual(state['iterations'], 0)


if __name__ == '__main__':
    unittest.main()

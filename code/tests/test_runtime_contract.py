"""Runtime output ownership and reviewer-facing builder contracts."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.options import args_parser

try:
    import torch
except ImportError:
    torch = None
else:
    import numpy as np
    import main_fed
    from builders import dataset_builder, model_builder
    from utils.state_dict_ops import model_param_dict


@unittest.skipIf(torch is None, 'requires Torch for real-model runtime checks')
class RuntimeContractTests(unittest.TestCase):
    def _args(self, *flags):
        with patch.object(sys, 'argv', ['test', *flags]):
            return args_parser()

    def test_existing_test_or_validation_trace_preserves_both_files(self):
        for protected in ('accuracy', 'validation'):
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as tmp:
                paths = {name: Path(tmp) / (name + '.txt') for name in ('accuracy', 'validation')}
                for name, path in paths.items():
                    path.write_text('existing result\n' if name == protected else '')
                before = {name: path.read_bytes() for name, path in paths.items()}
                with self.assertRaises(FileExistsError):
                    main_fed.reset_evaluation_outputs(str(paths['accuracy']), str(paths['validation']))
                self.assertEqual({name: path.read_bytes() for name, path in paths.items()}, before)

    def test_only_new_traces_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            accuracy, validation = Path(tmp) / 'accuracy.txt', Path(tmp) / 'validation.txt'
            main_fed.reset_evaluation_outputs(str(accuracy), str(validation))
            self.assertEqual(accuracy.read_bytes(), b'')
            self.assertEqual(validation.read_bytes(), b'')

    def test_empty_trace_is_reserved_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'accuracy.txt'
            path.touch()
            with self.assertRaises(FileExistsError):
                main_fed.reset_evaluation_outputs(str(path))
            self.assertEqual(path.read_bytes(), b'')

    def test_same_identity_concurrent_start_preserves_winners_results(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'accuracy.txt')
            ready = threading.Barrier(2)
            outcomes = []
            def start():
                ready.wait()
                try:
                    main_fed.reset_evaluation_outputs(path)
                except FileExistsError:
                    outcomes.append('rejected')
                else:
                    Path(path).write_text('50\n')
                    outcomes.append('owner')
            workers = [threading.Thread(target=start) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            self.assertCountEqual(outcomes, ['owner', 'rejected'])
            self.assertEqual(Path(path).read_text(), '50\n')

    def test_generated_adamasfl_seeds_have_separate_config_directories(self):
        import shlex
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
        import generate_missing_grid as grid
        directories = []
        for seed in (1, 2):
            command = shlex.split(grid._build_command('cifar', 'AdaMasFL', 0.1, 'pareto', 40, 5, seed))
            directories.append(command[command.index('--output_dir') + 1])
        self.assertNotEqual(*directories)
        self.assertTrue(all(path.endswith('/attempt=1') for path in directories))

    def test_main_rejects_existing_accuracy_before_local_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args('--output_dir', tmp, '--gpu', '-1')
            path = Path(main_fed.build_output_path(args))
            path.write_text('50\n')
            with patch.object(main_fed, 'args_parser', return_value=args), \
                    patch.object(main_fed, 'build_training_state') as build:
                with self.assertRaises(FileExistsError):
                    main_fed.main()
            build.assert_not_called()
            self.assertEqual(path.read_text(), '50\n')

    def test_default_paths_are_external_and_data_override_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'SAW_STATE_ROOT': str(Path(tmp) / 'runtime'),
            'SAW_DATA_ROOT': str(Path(tmp) / 'data'),
        }):
            args = self._args('--gpu', '-1')
            main_fed.configure_runtime(args)
            self.assertEqual(args.data_dir, str((Path(tmp) / 'data').resolve()))
            self.assertTrue(Path(main_fed.build_output_path(args)).is_relative_to((Path(tmp) / 'runtime').resolve()))
            args = self._args('--gpu', '-1', '--data_dir', str(Path(tmp) / 'override'))
            main_fed.configure_runtime(args)
            self.assertEqual(args.data_dir, str((Path(tmp) / 'override').resolve()))

    def test_repository_data_and_output_roots_are_rejected(self):
        repository = Path(main_fed.__file__).resolve().parents[1]
        args = self._args('--gpu', '-1', '--data_dir', str(repository / 'data'))
        with self.assertRaisesRegex(ValueError, 'outside the source repository'):
            main_fed.configure_runtime(args)
        args = self._args('--output_dir', str(repository / 'experiment_results'))
        with self.assertRaisesRegex(ValueError, 'outside the source repository'):
            main_fed.build_output_path(args)
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / 'source-alias'
            alias.symlink_to(repository, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'outside the source repository'):
                main_fed.external_path(alias / 'logs', 'logs')

    def test_existing_metrics_are_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args('--system_metrics_log_dir', tmp)
            output = str(Path(tmp) / 'run-test_acc.txt')
            metrics = Path(tmp) / 'run-system_metrics.csv'
            metrics.write_text('round,accuracy\n1,50\n')
            with self.assertRaises(FileExistsError):
                main_fed.build_system_metrics_path(args, output)
            self.assertEqual(metrics.read_text(), 'round,accuracy\n1,50\n')

    def test_five_data_builders_and_real_model_forwards(self):
        from PIL import Image
        from torchvision import datasets

        class InMemoryCifar(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.root, self.transform = root, transform
                self.targets = [0, 1, 0, 1]

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, index):
                image = Image.fromarray(np.full((32, 32, 3), index * 30, dtype=np.uint8))
                return self.transform(image), self.targets[index]

        settings = {'cifar': ('resnet', 10), 'cifar100': ('resnet', 100),
                    'tinyimagenet': ('resnet', 200), 'femnist': ('cnn', 62),
                    'gspeech': ('melcnn', 2)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for split in ('train', 'val'):
                for label in ('a', 'b'):
                    folder = root / 'tiny-imagenet-200' / split / label
                    folder.mkdir(parents=True)
                    for index in range(2):
                        Image.fromarray(np.full((64, 64, 3), index * 30, dtype=np.uint8)).save(folder / f'{index}.png')
            for dataset in ('femnist', 'gspeech'):
                data = {'user_data': {client: {
                    'x': [[0.0] * (784 if dataset == 'femnist' else 512)] * 2,
                    'y': [0, 1],
                } for client in ('client-b', 'client-a')}}
                for split in ('train', 'test'):
                    folder = root / dataset / split
                    folder.mkdir(parents=True)
                    (folder / 'data.json').write_text(json.dumps(data))
            for dataset, (model, classes) in settings.items():
                with self.subTest(dataset=dataset), contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(datasets, 'CIFAR10', InMemoryCifar))
                    stack.enter_context(patch.object(datasets, 'CIFAR100', InMemoryCifar))
                    stack.enter_context(patch.dict(os.environ, {'GSPEECH_FEATURE_CACHE': '0'}))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    args = self._args('--dataset', dataset, '--model', model, '--gpu', '-1',
                                      '--num_users', '2', '--distribution', 'iid', '--data_dir', tmp,
                                      '--femnist_min_samples_per_client', '0',
                                      '--gspeech_min_samples_per_client', '0')
                    main_fed.configure_runtime(args)
                    train, test, users, counts, metadata = dataset_builder.build_dataset(args)
                    self.assertEqual(len(train), 4)
                    self.assertEqual(len(test), 4)
                    self.assertEqual(set(users), {0, 1})
                    np.testing.assert_array_equal(counts, [2, 2])
                    self.assertTrue(Path(train.root).is_relative_to(root))
                    if dataset in ('femnist', 'gspeech'):
                        self.assertEqual(metadata['selected_client_ids'], ['client-a', 'client-b'])
                    images, targets = next(iter(torch.utils.data.DataLoader(train, batch_size=2)))
                    net, state = model_builder.build_model(args)
                    net.eval()
                    with torch.no_grad():
                        output = net(images)
                    self.assertEqual(tuple(output.shape), (2, classes))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(all(value.device.type == 'cpu' for value in state.values()))
                    self.assertFalse(any(name.startswith(('transformers', 'peft', 'data_reader.glue_text'))
                                         for name in sys.modules))

    def test_main_runs_one_commit_and_records_cache_identity_for_all_methods(self):
        from test_arrival_frozen_cache import CacheDeviceTests
        methods = CacheDeviceTests.METHODS
        dataset = torch.utils.data.TensorDataset(torch.tensor([[0.0, 1.0], [1.0, 0.0],
                                                               [1.0, 2.0], [2.0, 1.0],
                                                               [2.0, 3.0], [3.0, 2.0],
                                                               [3.0, 4.0], [4.0, 3.0]]),
                                                 torch.arange(8) % 2)
        users = {i: [2*i, 2*i+1] for i in range(4)}

        def model(args):
            net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
            return net, model_param_dict(net)

        for method in methods:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                flags = ['test', '--algo', method, '--gpu', '-1', '--output_dir', tmp,
                         '--num_users', '4', '--concurrency', '2', '--buffer_size', '1',
                         '--local_period', '1', '--local_bs', '2', '--bs', '2',
                         '--total_rounds', '1', '--lr', '0.02', '--global_lr', '0.1',
                         '--adamasfl_beta', '0.3', '--adamasfl_gamma', '0.1', '--adamasfl_lr', '0.02']
                if method == 'AdaMasFL':
                    flags.extend(['--adamasfl_cache_device', 'cpu'])
                else:
                    flags.extend(['--cache-device', 'cpu'])
                with patch.object(sys, 'argv', flags), \
                        patch.object(main_fed, 'build_dataset', return_value=(dataset, dataset, users, [2]*4, {})), \
                        patch.object(main_fed, 'build_model', side_effect=model), \
                        patch('algorithm.dispatcher._random_cost', return_value=lambda idx=None: float(idx + 1)), \
                        patch.dict(os.environ, {'SAW_SOURCE_COMMIT': '0fb63e6+dirty',
                                                'SAW_SOURCE_MANIFEST': 'snapshot-only'}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main_fed.main()
                accuracy = list(Path(tmp).glob('*-test_acc.txt'))
                self.assertEqual(len(accuracy), 1)
                values = np.loadtxt(accuracy[0], ndmin=1)
                self.assertEqual(values.shape, (1,))
                self.assertTrue(np.isfinite(values).all())
                configs = list(Path(tmp).glob('*resolved_config.json'))
                self.assertEqual(len(configs), 1)
                config = json.loads(configs[0].read_text())
                self.assertEqual(config['cache_device'], 'cpu')
                self.assertEqual(config['args']['cache_device'], 'cpu')
                self.assertEqual(config['source_commit'], '0fb63e6+dirty')
                self.assertEqual(config['source_manifest'], 'snapshot-only')
                identities = list(Path(tmp).glob('*-test_acc-identity.json'))
                self.assertEqual(len(identities), 1)
                identity = json.loads(identities[0].read_text())
                self.assertEqual(identity['args'], config['args'])
                self.assertEqual(identity['cache_device'], 'cpu')
                self.assertEqual(identity['source_commit'], '0fb63e6+dirty')
                self.assertEqual(identity['source_manifest'], 'snapshot-only')
                self.assertEqual(identity['status'], 'completed')
                self.assertEqual(identity['exit_code'], 0)
                self.assertEqual(identity['actual_command'], [sys.executable, *flags])


def fake_trainer_entrypoint():
    """Execute entrypoint functions with stdlib dependencies and a fake trainer."""
    source = Path(__file__).resolve().parents[1] / 'main_fed.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    stdlib = {'copy', 'gc', 'csv', 'json', 'math', 'os', 'random', 'time',
              'tracemalloc', 'resource', 'subprocess', 'sys', 'pathlib'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             or isinstance(node, ast.Import) and all(alias.name in stdlib for alias in node.names)
             or isinstance(node, ast.ImportFrom) and node.module in stdlib]
    namespace = {'__file__': str(source), 'args_parser': args_parser,
                 'torch': SimpleNamespace(device=type('FakeDevice', (), {})),
                 'resolve_cache_device': lambda args: args.adamasfl_cache_device or args.cache_device}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)

    def configure(args):
        args.device = 'cpu'
        args.data_dir = namespace['external_path'](args.data_dir or '/tmp/saw-fake-data', 'data_dir')

    def build(args):
        args.training_size_after_validation_split = 4
        state = {'iterations': 0, 'global_cost': 0, 'adamasfl_hparams': {},
                 'adamasfl_param_names': [], 'adamasfl_cache_device': 'cpu'}
        return state, [], [], {}, [], None

    def round_step(args, state, *unused):
        state['iterations'] += 1
        state['global_cost'] += 0.5
        return state

    namespace.update(configure_runtime=configure, build_training_state=build,
                     run_one_round=round_step, setup_direction_skew_monitor=lambda **kwargs: None,
                     log_direction_skew_metrics=lambda *args: None,
                     evaluate_global_model=lambda *args: (50.0, 0.5),
                     _current_device_memory_mb=lambda args: '')
    return namespace


class EntrypointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tpds-runtime-identity-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.entry = fake_trainer_entrypoint()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ('SAW_SOURCE_COMMIT', 'SAW_SOURCE_MANIFEST'):
            os.environ.pop(name, None)

    def argv(self, method='FedAsync', relative=False):
        root = os.path.relpath(self.root) if relative else str(self.root)
        script = ('code/main_fed.py' if relative else self.entry['__file__'])
        return [script, '--algo', method, '--dataset', 'cifar100', '--alpha', '0.1',
                '--num_users', '4', '--seed', '2', '--concurrency', '2', '--buffer_size', '1',
                '--total_rounds', '3', '--gpu', '-1', '--interval', '1',
                '--output_dir', root + '/active/seed=2/attempt=1/accuracy',
                '--system_metrics_log_dir', root + '/active/seed=2/attempt=1/metrics']

    def run_main(self, argv=None):
        with patch.object(sys, 'argv', argv or self.argv()), contextlib.redirect_stdout(io.StringIO()):
            self.entry['main']()

    def identity(self):
        paths = list(self.root.rglob('*-test_acc-identity.json'))
        self.assertEqual(len(paths), 1)
        return paths[0], json.loads(paths[0].read_text())

    def test_main_to_collector_preserves_actual_argv_and_complete_family(self):
        import gather
        for relative, method in ((False, 'FedAsync'), (True, 'AdaMasFL')):
            with self.subTest(relative=relative, method=method):
                self.root = Path(self.temporary.name).resolve() / method
                argv = self.argv(method, relative)
                self.run_main(argv)
                path, identity = self.identity()
                self.assertEqual(identity['argv'], argv)
                self.assertEqual(identity['actual_command'], [sys.executable, *argv])
                self.assertEqual(identity['command'], identity['actual_command'])
                self.assertEqual(identity['executable'], sys.executable)
                self.assertEqual(identity['cwd'], os.getcwd())
                self.assertEqual(identity['status'], 'completed')
                self.assertEqual(identity['exit_code'], 0)
                self.assertEqual(identity['completed_rounds'], 3)
                self.assertEqual(identity['attempt'], '1')
                self.assertEqual(identity['cache_device'], 'cpu')
                config = json.loads(Path(identity['resolved_config_path']).read_text())
                self.assertEqual(identity['args'], config['args'])
                self.assertEqual(identity['source_commit'], config['source_commit'])
                self.assertEqual(identity['code_identity'], config['code_identity'])
                self.assertIsNone(identity['source_manifest'])
                self.assertEqual(config['args']['training_size_after_validation_split'], 4)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = gather.main(['--source', str(self.root / 'active'), '--output-dir',
                                          str(self.root / 'review'), '--total-rounds', '3', '--copy'])
                self.assertEqual(result, 0)
                copied = self.root / 'review/awaiting_review' / path.relative_to(self.root / 'active')
                self.assertEqual(copied.read_bytes(), path.read_bytes())
                self.assertTrue(Path(identity['accuracy_path']).exists())

    def test_explicit_source_snapshot_is_preserved_without_inventing_manifest(self):
        with patch.dict(os.environ, {'SAW_SOURCE_COMMIT': '0fb63e6+dirty'}), \
                patch.object(subprocess, 'check_output', side_effect=AssertionError('unexpected Git probe')):
            self.run_main()
        _, identity = self.identity()
        self.assertEqual(identity['source_commit'], '0fb63e6+dirty')
        self.assertIsNone(identity['source_manifest'])
        self.assertEqual(identity['code_identity']['origin'], 'environment')

    def test_git_identity_detects_nested_modified_staged_and_untracked_code(self):
        repository = self.root / 'source'
        code = repository / 'code'
        nested = code / 'algorithm'
        nested.mkdir(parents=True)
        (code / 'main_fed.py').write_text('entrypoint')
        tracked = nested / 'trainer.py'
        tracked.write_text('original')
        def git(*tokens):
            return subprocess.check_output(['git', '-C', str(repository), *tokens], text=True).strip()
        git('init', '-q')
        git('add', 'code')
        git('-c', 'user.name=Runtime Test', '-c', 'user.email=runtime@example.invalid',
            'commit', '-qm', 'fixture')
        self.entry['__file__'] = str(code / 'main_fed.py')
        clean = self.entry['source_identity']()
        self.assertEqual(clean['source_commit'], git('rev-parse', 'HEAD'))
        self.assertFalse(clean['code_identity']['dirty'])
        tracked.write_text('modified')
        unstaged = self.entry['source_identity']()
        self.assertTrue(unstaged['source_commit'].endswith('+dirty'))
        self.assertIn('algorithm/trainer.py', str(unstaged['code_identity']['git_status']))
        git('add', 'code')
        (nested / 'new_trainer.py').write_text('new source')
        dirty = self.entry['source_identity']()
        self.assertEqual(dirty['source_commit'], clean['source_commit'] + '+dirty')
        self.assertTrue(any(line.startswith('M ') for line in dirty['code_identity']['git_status']))
        self.assertTrue(any('?? code/algorithm/new_trainer.py' == line
                            for line in dirty['code_identity']['git_status']))

    def test_no_git_snapshot_requires_explicit_source_before_outputs_are_reserved(self):
        with patch.object(subprocess, 'check_output', side_effect=FileNotFoundError('git')):
            with self.assertRaisesRegex(RuntimeError, 'provide SAW_SOURCE_COMMIT'):
                self.run_main()
        self.assertFalse(list(self.root.rglob('*.json')))
        self.assertFalse(list(self.root.rglob('*.txt')))

    def test_all_existing_result_members_reject_before_build_and_preserve_contents(self):
        argv = self.argv()
        with patch.object(sys, 'argv', argv):
            args = args_parser()
        accuracy = Path(self.entry['build_output_path'](args))
        protected = [accuracy, accuracy.with_name(accuracy.stem + '-identity.json'),
                     accuracy.with_name(accuracy.stem + '-resolved_config.json'),
                     Path(self.entry['system_metrics_output_path'](args, str(accuracy)))]
        for path in protected:
            with self.subTest(path=path.name):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('existing evidence')
                with patch.dict(self.entry, build_training_state=unittest.mock.Mock()):
                    build = self.entry['build_training_state']
                    with self.assertRaises(FileExistsError):
                        self.run_main(argv)
                    build.assert_not_called()
                self.assertEqual(path.read_text(), 'existing evidence')
                self.assertEqual(list(self.root.rglob('*.*')), [path])
                path.unlink()

    def test_existing_paired_symlink_is_never_followed(self):
        argv = self.argv()
        with patch.object(sys, 'argv', argv):
            args = args_parser()
        accuracy = Path(self.entry['build_output_path'](args))
        paired = accuracy.with_name(accuracy.stem + '-identity.json')
        target = self.root / 'external.json'
        target.write_text('protected')
        paired.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self.run_main(argv)
        self.assertTrue(paired.is_symlink())
        self.assertEqual(target.read_text(), 'protected')

    def test_controller_root_identity_is_preserved(self):
        controller = self.root / 'active/seed=2/attempt=1/identity.json'
        controller.parent.mkdir(parents=True)
        controller.write_text('{"controller_owned": true}')
        self.run_main()
        self.assertEqual(controller.read_text(), '{"controller_owned": true}')
        _, identity = self.identity()
        self.assertEqual(identity['exit_code'], 0)

    def test_build_and_training_errors_record_failed_identity_and_propagate_original(self):
        for phase in ('build_training_state', 'run_one_round'):
            with self.subTest(phase=phase):
                self.root = Path(self.temporary.name).resolve() / phase
                error = RuntimeError('fake trainer failed')
                def fail(*args):
                    raise error
                with patch.dict(self.entry, {phase: fail}):
                    with self.assertRaises(RuntimeError) as caught:
                        self.run_main()
                self.assertIs(caught.exception, error)
                _, identity = self.identity()
                self.assertEqual(identity['status'], 'failed')
                self.assertEqual(identity['exit_code'], 1)
                self.assertEqual(identity['error_type'], 'RuntimeError')

    def test_interrupt_and_nonzero_system_exit_are_recorded_and_propagated(self):
        for index, (error, expected) in enumerate(((KeyboardInterrupt(), 130), (SystemExit(7), 7),
                                                    (SystemExit(0), 0), (SystemExit(None), 0))):
            with self.subTest(error=type(error).__name__, code=expected):
                self.root = Path(self.temporary.name).resolve() / str(index)
                def fail(*args):
                    raise error
                with patch.dict(self.entry, run_one_round=fail):
                    with self.assertRaises(type(error)):
                        self.run_main()
                _, identity = self.identity()
                self.assertEqual(identity['status'], 'failed')
                self.assertEqual(identity['exit_code'], expected)

    def test_failure_recording_error_does_not_mask_trainer_traceback(self):
        original = RuntimeError('original training error')
        write = self.entry['write_owned_identity']
        def fail(*args):
            raise original
        def write_failure(handle, identity):
            if identity['status'] == 'failed':
                raise OSError('disk error')
            write(handle, identity)
        stderr = io.StringIO()
        with patch.dict(self.entry, run_one_round=fail, write_owned_identity=write_failure), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(RuntimeError) as caught:
                self.run_main()
        self.assertIs(caught.exception, original)
        self.assertIn('disk error', stderr.getvalue())

    def test_concurrent_main_invocations_have_one_paired_identity_owner(self):
        import threading
        ready = threading.Barrier(2)
        outcomes = []
        def start():
            ready.wait()
            try:
                self.entry['main']()
            except FileExistsError:
                outcomes.append('rejected')
            else:
                outcomes.append('owner')
        with patch.object(sys, 'argv', self.argv()), \
                patch.dict(os.environ, {'SAW_SOURCE_COMMIT': 'fixture-commit'}), \
                contextlib.redirect_stdout(io.StringIO()):
            workers = [threading.Thread(target=start) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        self.assertCountEqual(outcomes, ['owner', 'rejected'])
        _, identity = self.identity()
        self.assertEqual(identity['status'], 'completed')
        self.assertEqual(Path(identity['accuracy_path']).read_text(), '50.0\n' * 3)

    def test_replaced_paired_identity_cannot_be_overwritten_by_original_owner(self):
        path = self.root / 'identity.json'
        with path.open('x+', encoding='utf-8') as handle:
            self.entry['write_owned_identity'](handle, {'status': 'running'})
            path.rename(self.root / 'original.json')
            path.write_text('other owner')
            with self.assertRaises(FileExistsError):
                self.entry['write_owned_identity'](handle, {'status': 'completed'})
        self.assertEqual(path.read_text(), 'other owner')
        self.assertEqual(json.loads((self.root / 'original.json').read_text())['status'], 'running')

    def test_subprocess_failure_keeps_traceback_and_nonzero_process_exit(self):
        argv = self.argv()
        script = ("import sys; from test_runtime_contract import fake_trainer_entrypoint; "
                  "entry=fake_trainer_entrypoint(); "
                  "entry['run_one_round']=lambda *args: (_ for _ in ()).throw(RuntimeError('trainer boom')); "
                  "sys.argv=" + repr(argv) + "; entry['main']()")
        code = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(code), str(code / 'tests'))),
                           PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', script], env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Traceback (most recent call last)', result.stderr)
        self.assertIn('RuntimeError: trainer boom', result.stderr)
        _, identity = self.identity()
        self.assertEqual(identity['status'], 'failed')
        self.assertEqual(identity['exit_code'], result.returncode)


if __name__ == '__main__':
    unittest.main()

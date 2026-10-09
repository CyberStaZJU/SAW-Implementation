import ast
import importlib.util
import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[1]


class PreparationPathTests(unittest.TestCase):
    def run_tool(self, name, data_root, *args):
        env = dict(os.environ, SAW_DATA_ROOT=str(data_root), PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, str(CODE / 'tools' / name), *args],
                              env=env, capture_output=True, text=True)

    def test_femnist_subset_uses_external_default_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            payload = {'user_data': {'client': {'x': [[0]], 'y': [1]}}}
            for split in ('train', 'test'):
                directory = data / 'femnist' / split
                directory.mkdir(parents=True)
                (directory / 'input.json').write_text(json.dumps(payload))
            result = self.run_tool('prepare_femnist_truncated.py', data,
                                   '--num-users', '1', '--min-samples-per-client', '1')
            self.assertEqual(result.returncode, 0, result.stderr)
            for split in ('train', 'test'):
                output = data / 'femnist_truncated' / split / f'femnist_truncated_{split}.json'
                self.assertEqual(json.loads(output.read_text())['users'], ['client'])

    def test_gspeech_uses_external_default_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            raw = data / 'gspeech' / 'raw'
            (raw / 'yes').mkdir(parents=True)
            (raw / 'yes/client_nohash_0.wav').write_bytes(b'')
            (raw / 'testing_list.txt').write_text('')
            result = self.run_tool('prepare_gspeech.py', data)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = data / 'gspeech/train/gspeech_train.json'
            self.assertEqual(json.loads(output.read_text())['users'], ['client'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is required for conversion')
    def test_converter_uses_external_default_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            payload = {'users': ['client'], 'num_samples': [1],
                       'user_data': {'client': {'x': [[0] * 784], 'y': [1]}}}
            for split in ('train', 'test'):
                directory = data / 'femnist_truncated' / split
                directory.mkdir(parents=True)
                (directory / f'femnist_truncated_{split}.json').write_text(json.dumps(payload))
            result = self.run_tool('convert_femnist_to_pt.py', data)
            self.assertEqual(result.returncode, 0, result.stderr)
            for split in ('train', 'test'):
                output = data / 'femnist_truncated' / split / f'femnist_truncated_{split}.pt'
                self.assertGreater(output.stat().st_size, 0)

    def test_preparation_rejects_repository_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tool, argument in [('prepare_gspeech.py', '--root'),
                                   ('prepare_femnist_truncated.py', '--output')]:
                with self.subTest(tool=tool):
                    result = self.run_tool(tool, tmp, argument, str(CODE / 'data'))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('outside the repository', result.stderr)

    def test_full_femnist_refuses_existing_output_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'femnist'
            data.mkdir()
            marker = data / 'owned.json'
            marker.write_text('keep')
            env = dict(os.environ, SAW_FEMNIST_ROOT=str(data), SAW_LEAF_ROOT=str(Path(tmp) / 'leaf'))
            result = subprocess.run(['bash', str(CODE / 'scripts/prepare_femnist_full.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires a fresh', result.stderr)
            self.assertEqual(marker.read_text(), 'keep')
            self.assertFalse((Path(tmp) / 'leaf').exists())


class AblationPathTests(unittest.TestCase):
    def test_new_layout_and_historical_fallback(self):
        path = CODE / 'plot_ablation_60_10.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_seed_path')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = {'Path': Path, 'ROOT': root, 'ABLATION_DIR': root / 'cifar100_ablation_60_10',
                         'LEGACY_DIR': 'cifar100-alpha{alpha}-100users-60Mc-10Buffer'}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            resolve = namespace['_seed_path']
            filename = 'cifar100-SAW_EqualW-0.1-pareto-60-10-1-test_acc.txt'
            current = namespace['ABLATION_DIR'] / 'SAW_EqualW/alpha=0.1/seed=1' / filename
            legacy = namespace['ABLATION_DIR'] / 'SAW_EqualW/alpha0.1/seed1' / filename
            self.assertEqual(resolve('SAW_EqualW', '0.1', 1, True), current)
            legacy.parent.mkdir(parents=True)
            legacy.write_text('legacy')
            self.assertEqual(resolve('SAW_EqualW', '0.1', 1, True), legacy)
            current.parent.mkdir(parents=True)
            current.write_text('current')
            self.assertEqual(resolve('SAW_EqualW', '0.1', 1, True), current)
            self.assertEqual(resolve('SAW', '0.1', 1, False),
                             root / 'cifar100-alpha0.1-100users-60Mc-10Buffer/SAW/seed=1/cifar100-SAW-0.1-pareto-60-10-1-test_acc.txt')


if __name__ == '__main__':
    unittest.main()

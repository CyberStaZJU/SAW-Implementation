import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# NumPy's native modules must survive restoration of the renderer mock.
import numpy

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import generate_missing_grid as generator

# Validation tests never initialize a renderer or its disk cache.
with patch.dict('sys.modules', {'matplotlib': MagicMock(), 'matplotlib.pyplot': MagicMock()}):
    import plot_ablation_60_10 as plot
from utils.result_validation import (
    ExcessResultError,
    IncompleteResultError,
    ResultValidationError,
    load_accuracy,
    tail10_accuracy,
)


BAD_COUNTS = (499, 501, 548, 546, 536, 558)
BAD_ROWS = ('nan', 'inf', '-inf', '-0.1', '100.1', '50 60', 'bad', '', '# 50')


def accuracy_text(count=500, value='50'):
    return (value + '\n') * count


class MemoryTraceTests(unittest.TestCase):
    def setUp(self):
        self.traces = {}
        self.open_patch = patch('builtins.open', side_effect=self.open_trace)
        self.open_patch.start()
        self.addCleanup(self.open_patch.stop)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def open_trace(self, path, mode='r', **kwargs):
        self.assertEqual(mode, 'r', 'tests must not write trace files')
        key = str(path)
        if key not in self.traces:
            raise FileNotFoundError(key)
        text = self.traces[key]
        if isinstance(text, bytes):
            text = text.decode(kwargs.get('encoding', 'utf-8'))
        return io.StringIO(text)

    def add_trace(self, name, text):
        path = Path(name)
        self.traces[str(path)] = text
        return path


class ResultValidationTests(MemoryTraceTests):
    def test_accepts_exact_count_and_accuracy_boundaries(self):
        path = self.add_trace('accuracy.txt', '0\n100\n' * 250)
        values = load_accuracy(path)
        self.assertEqual(len(values), 500)
        self.assertEqual(values[:2], [0, 100])

    def test_expected_count_is_configurable(self):
        path = self.add_trace('accuracy.txt', ' 0 \n50\n100')
        self.assertEqual(load_accuracy(path, expected_rounds=3), [0, 50, 100])

    def test_rejects_short_and_excess_counts(self):
        for count in BAD_COUNTS:
            with self.subTest(count=count):
                path = self.add_trace('accuracy.txt', accuracy_text(count))
                error = IncompleteResultError if count < 500 else ExcessResultError
                with self.assertRaises(error) as caught:
                    load_accuracy(path)
                self.assertIn(str(count), str(caught.exception))
                self.assertIn('500', str(caught.exception))
                self.assertIn(str(path), str(caught.exception))

    def test_rejects_invalid_rows_with_line_number(self):
        for row in BAD_ROWS:
            with self.subTest(row=row):
                path = self.add_trace('accuracy.txt', accuracy_text(20) + row + '\n' + accuracy_text(479))
                with self.assertRaises(ResultValidationError) as caught:
                    load_accuracy(path)
                self.assertNotIsInstance(caught.exception, IncompleteResultError)
                self.assertIn('line 21', str(caught.exception))

    def test_rejects_bad_encoding(self):
        path = self.add_trace('accuracy.txt', b'\xff\n')
        with self.assertRaisesRegex(ResultValidationError, 'UTF-8'):
            load_accuracy(path)

    def test_short_malformed_trace_is_invalid_not_incomplete(self):
        path = self.add_trace('accuracy.txt', 'nan\n')
        with self.assertRaises(ResultValidationError) as caught:
            load_accuracy(path)
        self.assertNotIsInstance(caught.exception, IncompleteResultError)

    def test_rejects_invalid_expected_count(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'positive integer'):
                load_accuracy('accuracy.txt', expected_rounds=count)

    def test_tail10_uses_rows_491_through_500(self):
        path = self.add_trace('accuracy.txt', accuracy_text(490, '99') + ''.join(f'{n}\n' for n in range(10)))
        self.assertEqual(tail10_accuracy(path), 4.5)

    def test_tail10_validates_rows_before_the_tail(self):
        path = self.add_trace('accuracy.txt', 'nan\n' + accuracy_text(499))
        with self.assertRaises(ResultValidationError):
            tail10_accuracy(path)

    def test_tail10_rejects_excess_instead_of_slicing(self):
        path = self.add_trace('accuracy.txt', accuracy_text(558))
        with self.assertRaises(ExcessResultError):
            tail10_accuracy(path)

    def test_tail10_requires_ten_evaluations(self):
        path = self.add_trace('accuracy.txt', accuracy_text(9))
        with self.assertRaisesRegex(ValueError, 'at least ten'):
            tail10_accuracy(path, expected_rounds=9)


class ConsumerValidationTests(MemoryTraceTests):
    def scan_generator(self, paths, expected=500):
        with patch.object(generator, 'MIN_RESULT_LINES', expected), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(Path, 'rglob', return_value=paths):
            return generator._collect_existing_result_names(['results'])

    def test_generator_accepts_valid_and_skips_only_valid_short(self):
        complete = self.add_trace('complete-test_acc.txt', accuracy_text())
        short = self.add_trace('short-test_acc.txt', accuracy_text(499))
        self.assertEqual(self.scan_generator([short, complete]), {complete.name})

    def run_generator_main(self, path, expected=500):
        with patch.object(generator, 'MIN_RESULT_LINES', expected), \
                patch.object(generator, 'RESULT_DIRS', ['results']), \
                patch.object(generator, 'DATASET_ALPHAS', {'cifar100': [0.1]}), \
                patch.object(generator, 'PAIRS', [(80, 10)]), \
                patch.object(generator, 'SEEDS', [2]), patch.object(generator, 'ALGOS', ['FedAsync']), \
                patch.object(Path, 'exists', return_value=True), patch.object(Path, 'rglob', return_value=[path]), \
                patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write, \
                patch.object(generator, '_build_command', return_value='training command') as command:
            try:
                generator.main()
            except ResultValidationError:
                write.assert_not_called()
                command.assert_not_called()
                raise
            return write, command

    def test_generator_blocks_bad_traces_before_building_or_writing_queue(self):
        texts = [accuracy_text(n) for n in BAD_COUNTS if n > 500]
        texts += [accuracy_text(499) + row + '\n' for row in BAD_ROWS]
        texts += ['nan\n', b'\xff\n']
        for text in texts:
            with self.subTest(text_length=len(text)):
                path = self.add_trace('cifar100-FedAsync-0.1-pareto-80-10-2-test_acc.txt', text)
                with self.assertRaises(ResultValidationError):
                    self.run_generator_main(path)

    def test_generator_emits_missing_task_for_valid_short_trace(self):
        path = self.add_trace('cifar100-FedAsync-0.1-pareto-80-10-2-test_acc.txt', accuracy_text(499))
        write, command = self.run_generator_main(path)
        command.assert_called_once_with('cifar100', 'FedAsync', 0.1, 'pareto', 80, 10, 2)
        write.assert_called_once()
        self.assertIn('exact_scalar_count expected_rounds=500', write.call_args.args[0])

    def test_generator_does_not_emit_task_for_valid_complete_trace(self):
        path = self.add_trace('cifar100-FedAsync-0.1-pareto-80-10-2-test_acc.txt', accuracy_text())
        write, command = self.run_generator_main(path)
        command.assert_not_called()
        self.assertIn('existing=1 missing=0', write.call_args.args[0])

    def test_generator_existence_only_mode_does_not_claim_completeness(self):
        path = self.add_trace('cifar100-FedAsync-0.1-pareto-80-10-2-test_acc.txt', 'nan\n')
        for expected in (0, -1):
            with self.subTest(expected=expected):
                self.assertEqual(self.scan_generator([path], expected), {path.name})
                write, command = self.run_generator_main(path, expected)
                command.assert_not_called()
                self.assertIn('existence_only; completeness not checked', write.call_args.args[0])

    def plot_traces(self):
        paths = {}
        for seed in plot.SEEDS:
            paths[seed] = self.add_trace(f'seed{seed}-test_acc.txt', accuracy_text(500, str(seed)))
        return paths

    def test_plot_loads_all_five_valid_seeds(self):
        paths = self.plot_traces()
        with patch.object(plot, '_seed_path', side_effect=lambda method, alpha, seed, ablation: paths[seed]):
            curves = plot._load_curves('SAW', '0.1', False)
        self.assertEqual(curves.shape, (5, 500))
        self.assertEqual(curves[:, -1].tolist(), [1, 2, 3, 4, 5])

    def test_plot_rejects_short_excess_and_invalid_values(self):
        paths = self.plot_traces()
        texts = [accuracy_text(n) for n in BAD_COUNTS] + [accuracy_text(499) + row + '\n' for row in BAD_ROWS]
        for text in texts:
            with self.subTest(text_length=len(text)):
                self.traces[str(paths[5])] = text
                with patch.object(plot, '_seed_path', side_effect=lambda method, alpha, seed, ablation: paths[seed]):
                    with self.assertRaises(ResultValidationError):
                        plot._load_curves('SAW', '0.1', False)

    def test_plot_rejects_missing_seed(self):
        paths = self.plot_traces()
        del self.traces[str(paths[5])]
        with patch.object(plot, '_seed_path', side_effect=lambda method, alpha, seed, ablation: paths[seed]):
            with self.assertRaisesRegex(FileNotFoundError, 'seed5'):
                plot._load_curves('SAW', '0.1', False)

    def test_plot_main_preflights_all_data_before_creating_figure(self):
        panel_count = len(plot.ALPHAS) * len(plot.METHODS)
        valid = plot.np.zeros((5, 500))
        results = [valid] * (panel_count - 1) + [ExcessResultError('bad final trace')]
        with patch.object(plot, '_load_curves', side_effect=results) as loader, \
                patch.object(plot.plt, 'subplots') as subplots:
            with self.assertRaises(ExcessResultError):
                plot.main()
        self.assertEqual(loader.call_count, panel_count)
        subplots.assert_not_called()


if __name__ == '__main__':
    unittest.main()

"""Stdlib-only collection contract; all artifacts live in temporary directories."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gather
import organize_completed_results as organizer


NAME = 'cifar100-AdaMasFL-0.1-pareto-40-5-2-test_acc.txt'
TOOLS = ('gather', 'organizer')


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tpds-collection-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / 'active'
        self.source.mkdir()

    def family(self, attempt=1, rounds=3, flat=False, source=None):
        root = (source or self.source) / 'task/seed=2' / f'attempt={attempt}'
        accuracy_dir = root if flat else root / 'accuracy'
        metrics_dir = root if flat else root / 'metrics'
        accuracy_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        accuracy = accuracy_dir / NAME
        accuracy.write_text('50\n' * rounds)
        args = {'algo': 'AdaMasFL', 'dataset': 'cifar100', 'alpha': 0.1, 'seed': 2,
                'num_users': 100, 'concurrency': 40, 'buffer_size': 5,
                'random_cost': 'pareto', 'total_rounds': rounds, 'interval': 1,
                'distribution': 'noniid', 'run_tag': '', 'output_dir': str(accuracy_dir),
                'system_metrics_log_dir': str(metrics_dir)}
        resolved = accuracy_dir / ('resolved_config.json' if not flat else
                                   accuracy.stem + '-resolved_config.json')
        resolved.write_text(json.dumps({'args': args, 'source_commit': 'fixture-commit'}))
        identity = root / 'identity.json'
        identity.write_text(json.dumps({
            'method': 'AdaMasFL', 'task': {'dataset': 'cifar100', 'alpha': 0.1},
            'config': {'num_users': 100, 'concurrency': 40, 'buffer': 5},
            'seed': 2, 'attempt': attempt, 'code_identity': {'commit': 'fixture-commit'},
            'command': ['python', 'main_fed.py'] + [token for field, value in args.items()
                                                  for token in ('--' + field, str(value))],
        }))
        metrics = metrics_dir / NAME.replace('-test_acc.txt', '-system_metrics.csv')
        fields = ['round', 'algo', 'dataset', 'alpha', 'random_cost', 'concurrency',
                  'buffer_size', 'seed', 'run_tag', *gather.METRIC_NUMBERS]
        with metrics.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for number in range(1, rounds + 1):
                writer.writerow({field: (number if field == 'round' else
                                         args.get(field, 1.0)) for field in fields})
        return {'root': root, 'accuracy': accuracy, 'resolved': resolved,
                'identity': identity, 'metrics': metrics}

    def run_tool(self, tool, *flags, source=None, target=None, rounds=3):
        source = source or self.source
        target = target or self.root / 'collected' / tool
        argv = (['--source', str(source), '--output-dir', str(target)] if tool == 'gather'
                else ['--sources', str(source), '--target-root', str(target)])
        if rounds is not None:
            argv += ['--total-rounds', str(rounds)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = (gather.main if tool == 'gather' else organizer.main)(argv + list(flags))
        return result, output.getvalue(), target / 'awaiting_review'

    def assert_rejected(self, *flags, source=None, rounds=3):
        before = tree_bytes(self.source)
        for tool in TOOLS:
            with self.subTest(tool=tool):
                result, output, target = self.run_tool(tool, '--copy', *flags,
                                                       source=source, rounds=rounds)
                self.assertEqual(result, 1, output)
                self.assertIn('reject batch:', output)
                self.assertFalse(target.parent.exists())
        self.assertEqual(tree_bytes(self.source), before)

    def test_unavailable_device_memory_is_preserved_as_blank(self):
        family = self.family()
        with family['metrics'].open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        with family['metrics'].open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row['device_peak_memory_mb'] = ''
                writer.writerow(row)
        original = family['metrics'].read_bytes()
        for tool in TOOLS:
            result, output, target = self.run_tool(tool, '--copy')
            self.assertEqual(result, 0, output)
            copied = target / family['metrics'].relative_to(self.source)
            self.assertEqual(copied.read_bytes(), original)

    def test_default_dry_run_makes_no_directories(self):
        self.family()
        before = tree_bytes(self.source)
        for tool in TOOLS:
            with self.subTest(tool=tool):
                result, output, target = self.run_tool(tool)
                self.assertEqual(result, 0, output)
                self.assertIn('mode=dry-run', output)
                self.assertIn('status=awaiting_review', output)
                self.assertFalse(target.parent.exists())
        self.assertEqual(tree_bytes(self.source), before)

    def test_copy_preserves_two_same_basenames_and_all_evidence(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                source = self.source / str(flat)
                for attempt in (1, 2):
                    family = self.family(attempt, flat=flat, source=source)
                    (family['root'] / 'audit.json').write_text('{"pending": true}')
                    (family['root'] / 'completion.json').write_text('{"exit_code": null}')
                    (family['root'] / 'trainer.log').write_text('necessary evidence\n')
                    family['accuracy'].with_name(NAME.replace('-test_acc.txt', '-validation_acc.txt')).write_text('49\n' * 3)
                    family['accuracy'].with_name(NAME.replace('-test_acc.txt', '-checkpoint.pt')).write_bytes(b'checkpoint')
                before = tree_bytes(source)
                expected = {name: content for name, content in before.items() if not name.endswith('.pt')}
                for tool in TOOLS:
                    result, output, target = self.run_tool(
                        tool, '--copy', source=source, target=self.root / f'{tool}-{flat}')
                    self.assertEqual(result, 0, output)
                    self.assertEqual(tree_bytes(target), expected)
                    self.assertEqual(tree_bytes(source), before)
                    self.assertEqual(len(list(target.rglob('*-test_acc.txt'))), 2)

    def test_active_500_rows_only_collects_to_awaiting_review(self):
        self.family(rounds=500)
        for tool in TOOLS:
            result, output, target = self.run_tool(tool, '--copy', rounds=None)
            self.assertEqual(result, 0, output)
            self.assertTrue(target.is_dir())
            self.assertIn('formal audit still required', output)
            self.assertFalse(list(self.root.rglob('verified')))

    def test_conflict_in_last_attempt_prevents_entire_batch(self):
        self.family(1)
        last = self.family(2)
        before_source = tree_bytes(self.source)
        for tool in TOOLS:
            target_root = self.root / 'collected' / tool
            target = target_root / 'awaiting_review'
            conflict = target / last['identity'].relative_to(self.source)
            conflict.parent.mkdir(parents=True)
            conflict.write_text('existing evidence')
            before_target = tree_bytes(target)
            with patch.object(gather, 'copy_plan') as copier:
                result, output, _ = self.run_tool(tool, '--copy', target=target_root)
            self.assertEqual(result, 1, output)
            copier.assert_not_called()
            self.assertEqual(tree_bytes(target), before_target)
            self.assertEqual(tree_bytes(self.source), before_source)

    def test_dry_run_also_reports_conflict(self):
        family = self.family()
        for tool in TOOLS:
            target_root = self.root / tool
            target = target_root / 'awaiting_review'
            conflict = target / family['metrics'].relative_to(self.source)
            conflict.parent.mkdir(parents=True)
            conflict.touch()
            result, output, _ = self.run_tool(tool, target=target_root)
            self.assertEqual(result, 1, output)
            self.assertEqual(len(tree_bytes(target)), 1)

    def test_destination_parent_file_is_preflighted(self):
        self.family()
        for tool in TOOLS:
            target_root = self.root / tool
            target_root.mkdir()
            (target_root / 'awaiting_review').write_text('not a directory')
            result, output, _ = self.run_tool(tool, '--copy', target=target_root)
            self.assertEqual(result, 1, output)
            self.assertEqual((target_root / 'awaiting_review').read_text(), 'not a directory')

    def test_missing_any_required_companion_rejects_batch(self):
        for missing in ('resolved', 'metrics', 'identity'):
            with self.subTest(missing=missing):
                family = self.family()
                family[missing].unlink()
                self.assert_rejected()

    def test_invalid_last_family_prevents_copy_of_first(self):
        self.family(1)
        self.family(2)['identity'].unlink()
        self.assert_rejected()

    def test_generic_config_cannot_be_shared_by_multiple_accuracies(self):
        family = self.family()
        family['accuracy'].with_name(NAME.replace('-2-test_acc', '-3-test_acc')).write_text('50\n' * 3)
        self.assert_rejected()

    def test_ambiguous_config_rejects(self):
        family = self.family()
        family['accuracy'].with_name(family['accuracy'].stem + '-resolved_config.json').write_bytes(
            family['resolved'].read_bytes())
        self.assert_rejected()

    def test_ambiguous_metrics_rejects(self):
        family = self.family()
        family['accuracy'].with_name(family['metrics'].name).write_bytes(family['metrics'].read_bytes())
        self.assert_rejected()

    def test_ambiguous_identity_rejects(self):
        family = self.family()
        family['accuracy'].with_name(family['accuracy'].stem + '-identity.json').write_bytes(
            family['identity'].read_bytes())
        self.assert_rejected()

    def test_trainer_identity_and_scheduler_identity_are_both_collected(self):
        family = self.family()
        paired = family['accuracy'].with_name(family['accuracy'].stem + '-identity.json')
        resolved = json.loads(family['resolved'].read_text())
        paired.write_text(json.dumps({**resolved, 'producer': 'main_fed', 'attempt': 1}))
        before_source = tree_bytes(self.source)
        for tool in TOOLS:
            result, output, target = self.run_tool(tool, '--copy')
            self.assertEqual(result, 0, output)
            for identity_path in (family['identity'], paired):
                copied = target / identity_path.relative_to(self.source)
                self.assertEqual(copied.read_bytes(), identity_path.read_bytes())
        self.assertEqual(tree_bytes(self.source), before_source)

    def test_conflicting_scheduler_and_trainer_identity_rejects_batch(self):
        family = self.family()
        paired = family['accuracy'].with_name(family['accuracy'].stem + '-identity.json')
        resolved = json.loads(family['resolved'].read_text())
        paired.write_text(json.dumps({**resolved, 'producer': 'main_fed', 'attempt': 1}))
        root_identity = json.loads(family['identity'].read_text())
        root_identity['seed'] = 3
        family['identity'].write_text(json.dumps(root_identity))
        self.assert_rejected()

    def test_full_trainer_args_and_explicit_scheduler_configuration_conflicts_reject(self):
        for field, value, wrong in [('model', 'resnet', 'melcnn'), ('lr', 0.01, 42.0),
                                    ('cache_device', 'cpu', 'device'),
                                    ('adamasfl_cache_device', 'cpu', 'device'),
                                    ('adamasfl_beta', 0.2, 0.7)]:
            for record_type in ('trainer_args', 'scheduler_command'):
                with self.subTest(field=field, record_type=record_type):
                    family = self.family()
                    resolved = json.loads(family['resolved'].read_text())
                    resolved['args'][field] = value
                    family['resolved'].write_text(json.dumps(resolved))
                    paired = family['accuracy'].with_name(family['accuracy'].stem + '-identity.json')
                    trainer = {**resolved, 'args': dict(resolved['args']),
                               'producer': 'main_fed', 'attempt': 1}
                    if record_type == 'trainer_args':
                        trainer['args'][field] = wrong
                    else:
                        scheduler = json.loads(family['identity'].read_text())
                        scheduler['command'] += ['--' + field, str(wrong)]
                        family['identity'].write_text(json.dumps(scheduler))
                    paired.write_text(json.dumps(trainer))
                    self.assert_rejected()

    def test_missing_mismatched_or_unrecorded_commit_rejects(self):
        for commit in (None, '', 'unrecorded', 'different-commit'):
            with self.subTest(commit=commit):
                family = self.family()
                value = json.loads(family['resolved'].read_text())
                value['source_commit'] = commit
                family['resolved'].write_text(json.dumps(value))
                self.assert_rejected()

    def test_mismatched_attempt_identity_fields_reject(self):
        mutations = ({'seed': 3}, {'attempt': 2}, {'method': 'SAW'},
                     {'task': {'dataset': 'cifar', 'alpha': 0.1}},
                     {'config': {'num_users': 50, 'concurrency': 40, 'buffer': 5}},
                     {'command': ['python', '--total_rounds', '9']},
                     {'source_manifest': 'different-manifest'})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                family = self.family()
                identity = json.loads(family['identity'].read_text())
                identity.update(mutation)
                family['identity'].write_text(json.dumps(identity))
                self.assert_rejected()

    def test_filename_must_match_config_identity(self):
        family = self.family()
        for key in ('accuracy', 'metrics'):
            family[key].rename(family[key].with_name(family[key].name.replace('AdaMasFL', 'SAW')))
        self.assert_rejected()

    def test_malformed_identity_or_config_rejects(self):
        for member in ('identity', 'resolved'):
            for text in ('[]', '{', '{"args": null}'):
                with self.subTest(member=member, text=text):
                    self.family()[member].write_text(text)
                    self.assert_rejected()

    def test_resolved_args_identity_format_is_supported(self):
        family = self.family(flat=True)
        resolved = json.loads(family['resolved'].read_text())
        family['identity'].write_text(json.dumps({**resolved, 'attempt': 1}))
        for tool in TOOLS:
            result, output, _ = self.run_tool(tool, '--copy')
            self.assertEqual(result, 0, output)

    def test_configured_final_round_must_match_request(self):
        self.family(rounds=4)
        self.assert_rejected()

    def test_accuracy_count_and_values_reject(self):
        texts = ['50\n' * n for n in (499, 501, 548, 546, 536, 558)]
        texts += ['50\n' * 499 + row + '\n'
                  for row in ('nan', 'inf', '-inf', '-0.1', '100.1', '50 60', 'bad', '')]
        for text in texts:
            with self.subTest(length=len(text)):
                self.family(rounds=500)['accuracy'].write_text(text)
                self.assert_rejected(rounds=500)

    def test_metrics_rounds_identity_and_finite_values_reject(self):
        for field, value in (('round', '2'), ('seed', '3'), ('dataset', 'cifar'),
                             ('alpha', '0.2'), ('run_tag', 'wrong'),
                             ('runtime_sec', 'nan'), ('process_memory_mb', '-1')):
            with self.subTest(field=field):
                family = self.family()
                with family['metrics'].open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    fields, rows = reader.fieldnames, list(reader)
                rows[0][field] = value
                with family['metrics'].open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                self.assert_rejected()

    def test_missing_short_excess_or_malformed_metric_rows_reject(self):
        for mutation in ('header', 'short', 'excess', 'ragged'):
            with self.subTest(mutation=mutation):
                family = self.family()
                lines = family['metrics'].read_text().splitlines(keepends=True)
                if mutation == 'header':
                    lines[0] = lines[0].replace('runtime_sec,', 'other,', 1)
                elif mutation == 'short':
                    lines.pop()
                elif mutation == 'excess':
                    lines.append(lines[-1])
                else:
                    lines[-1] = '3\n'
                family['metrics'].write_text(''.join(lines))
                self.assert_rejected()

    def test_evaluation_interval_uses_config_with_all_metric_rounds(self):
        family = self.family(rounds=4)
        resolved = json.loads(family['resolved'].read_text())
        identity = json.loads(family['identity'].read_text())
        resolved['args']['interval'] = 2
        identity['command'][identity['command'].index('--interval') + 1] = '2'
        family['resolved'].write_text(json.dumps(resolved))
        family['identity'].write_text(json.dumps(identity))
        family['accuracy'].write_text('50\n' * 2)
        for tool in TOOLS:
            result, output, _ = self.run_tool(tool, '--copy', rounds=4)
            self.assertEqual(result, 0, output)

    def test_symlinked_evidence_rejects(self):
        family = self.family()
        original = family['identity'].read_bytes()
        family['identity'].unlink()
        external = self.root / 'external-identity.json'
        external.write_bytes(original)
        family['identity'].symlink_to(external)
        self.assert_rejected()

    def test_accuracy_only_source_directory_cannot_split_attempt(self):
        family = self.family()
        self.assert_rejected(source=family['accuracy'].parent)

    def test_single_accuracy_file_preserves_attempt_family(self):
        family = self.family()
        for tool in TOOLS:
            result, output, target = self.run_tool(tool, '--copy', source=family['accuracy'])
            self.assertEqual(result, 0, output)
            self.assertEqual(tree_bytes(target / 'attempt=1'), tree_bytes(family['root']))

    def test_multisource_collision_rejects_before_any_copy(self):
        first, second = self.root / 'first', self.root / 'second'
        self.family(source=first)
        self.family(source=second)
        target = self.root / 'awaiting_review'
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(gather, 'copy_plan') as copier:
            result = organizer.main(['--sources', str(first), str(second),
                                     '--target-root', str(target), '--copy', '--total-rounds', '3'])
        self.assertEqual(result, 1, output.getvalue())
        copier.assert_not_called()
        self.assertFalse(target.exists())

    def test_copy_error_rolls_back_only_new_files_and_directories(self):
        self.family()
        before = tree_bytes(self.source)
        for tool in TOOLS:
            original = gather.shutil.copyfileobj
            calls = []
            def failing_copy(source, target):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError('injected copy failure')
                original(source, target)
            with patch.object(gather.shutil, 'copyfileobj', side_effect=failing_copy):
                result, output, target = self.run_tool(tool, '--copy')
            self.assertEqual(result, 1, output)
            self.assertFalse(target.parent.exists())
            self.assertEqual(tree_bytes(self.source), before)

    def test_source_change_during_copy_rolls_back(self):
        family = self.family()
        original = gather.shutil.copyfileobj
        def mutate_source(source, target):
            original(source, target)
            with family['accuracy'].open('a') as handle:
                handle.write('51\n')
        with patch.object(gather.shutil, 'copyfileobj', side_effect=mutate_source):
            result, output, target = self.run_tool('gather', '--copy')
        self.assertEqual(result, 1, output)
        self.assertIn('evidence changed while copying', output)
        self.assertFalse(target.parent.exists())
        self.assertTrue(family['accuracy'].exists())

    def test_filename_alpha_must_match_config(self):
        family = self.family()
        for key in ('accuracy', 'metrics'):
            family[key].rename(family[key].with_name(family[key].name.replace('-0.1-', '-0.2-')))
        self.assert_rejected()

    def test_destination_conflict_created_after_preflight_rolls_back_new_files(self):
        self.family()
        target = self.root / 'awaiting_review'
        plan = gather.preflight([self.source], target, 3)
        conflict = plan[-1].destination
        conflict.parent.mkdir(parents=True)
        conflict.write_text('concurrent result')
        before = tree_bytes(target)
        with self.assertRaises(FileExistsError):
            gather.copy_plan(plan)
        self.assertEqual(tree_bytes(target), before)
        self.assertEqual(conflict.read_text(), 'concurrent result')

    def test_source_change_after_preflight_writes_nothing(self):
        family = self.family()
        target = self.root / 'awaiting_review'
        plan = gather.preflight([self.source], target, 3)
        family['accuracy'].write_text('51\n' * 4)
        with self.assertRaisesRegex(ValueError, 'evidence changed after preflight'):
            gather.copy_plan(plan)
        self.assertFalse(target.exists())

    def test_verified_repository_and_overlapping_destinations_reject(self):
        self.family()
        for tool in TOOLS:
            targets = [self.root / 'runs/verified', Path(gather.__file__).resolve().parent / 'awaiting_review',
                       self.source / 'awaiting_review']
            for target in targets:
                with self.subTest(tool=tool, target=target):
                    result, output, _ = self.run_tool(tool, '--copy', target=target)
                    self.assertEqual(result, 1, output)
                    self.assertFalse(target.exists())

    def test_non_accuracy_file_source_is_rejected(self):
        family = self.family()
        self.assert_rejected(source=family['metrics'])

    def test_missing_source_rejects_without_creating_output(self):
        self.assert_rejected(source=self.root / 'missing')

    def test_explicit_dry_run_and_min_lines_alias(self):
        self.family()
        for tool in TOOLS:
            result, output, target = self.run_tool(tool, '--dry-run', '--min-lines', '3', rounds=None)
            self.assertEqual(result, 0, output)
            self.assertFalse(target.parent.exists())


if __name__ == '__main__':
    unittest.main()

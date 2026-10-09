"""Validate scalar accuracy structure; round provenance needs a separate audit."""

import math
from os import PathLike


class ResultValidationError(ValueError):
    """An accuracy trace has invalid scalar structure or values."""


class IncompleteResultError(ResultValidationError):
    """A structurally valid trace has fewer evaluations than expected."""


class ExcessResultError(ResultValidationError):
    """A trace has more evaluations than expected."""


def load_accuracy(path: str | PathLike[str], expected_rounds: int = 500) -> list[float]:
    """Require exactly expected_rounds lines, each a finite scalar in [0, 100].

    Passing this check does not prove round identity or continuity. Historical
    scalar traces still need an independent round/log provenance audit.
    """
    if not isinstance(expected_rounds, int) or isinstance(expected_rounds, bool) or expected_rounds <= 0:
        raise ValueError('expected_rounds must be a positive integer')

    values = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                tokens = line.split()
                if len(tokens) != 1:
                    raise ResultValidationError(
                        f'{path}: line {line_number}: expected one scalar accuracy value'
                    )
                try:
                    value = float(tokens[0])
                except ValueError as exc:
                    raise ResultValidationError(
                        f'{path}: line {line_number}: invalid accuracy value {tokens[0]!r}'
                    ) from exc
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ResultValidationError(
                        f'{path}: line {line_number}: accuracy must be finite and in [0, 100]'
                    )
                values.append(value)
    except UnicodeError as exc:
        raise ResultValidationError(f'{path}: invalid UTF-8 accuracy text') from exc

    if len(values) < expected_rounds:
        raise IncompleteResultError(
            f'{path}: incomplete accuracy trace: expected {expected_rounds} rows, found {len(values)}'
        )
    if len(values) > expected_rounds:
        raise ExcessResultError(
            f'{path}: excess accuracy rows: expected {expected_rounds}, found {len(values)}'
        )
    return values


def tail10_accuracy(path: str | PathLike[str], expected_rounds: int = 500) -> float:
    """Validate the full trace before averaging its final ten scalar rows.

    For an independently audited 500-round trace, these are rounds 491–500.
    Scalar row count alone does not establish that round provenance.
    """
    values = load_accuracy(path, expected_rounds)
    if expected_rounds < 10:
        raise ValueError('Tail-10 requires at least ten expected rounds')
    return math.fsum(values[-10:]) / 10

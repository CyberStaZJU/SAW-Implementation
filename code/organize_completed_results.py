#!/usr/bin/env python3
"""Preview or copy complete result families into external awaiting_review.

Use the same family validation and batch preflight as gather.py. Paths stay
relative to each source root; taxonomy is no longer inferred from filenames.
Collection does not certify completion or promote a run to verified results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gather import INPUT_DIR, MIN_LINES, OUTPUT_DIR, collect_results


def collect_completed(args: argparse.Namespace) -> int:
    return collect_results([Path(item) for item in args.sources], Path(args.target_root),
                           args.total_rounds, args.copy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=[INPUT_DIR],
                        help='recursive roots or accuracy files; include complete attempts')
    parser.add_argument('--target-root', default=OUTPUT_DIR,
                        help='external review root; awaiting_review is appended if absent')
    parser.add_argument('--total-rounds', '--min-lines', type=int, default=MIN_LINES,
                        help='required configured final round (default: 500)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--copy', action='store_true', help='copy families; preserve all originals')
    mode.add_argument('--dry-run', action='store_true', help='preview only (the default)')
    args = parser.parse_args(argv)
    try:
        return collect_completed(args)
    except (OSError, ValueError, TypeError) as exc:
        print(f'reject batch: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

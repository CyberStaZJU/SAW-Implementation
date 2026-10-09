#!/usr/bin/env python3
"""Regenerate paper/ablation_figure_abc.png for the (Mc, B) = (60, 10) setting.

Three panels (a/b/c) = alpha in {0.01, 0.1, 0.5}.
Each panel: test-accuracy trajectories for SAW, SAW_EqualW, SAW_AdaInside, FedBuff
across 5 seeds, smoothed with moving-average window 10, with 95% bootstrap CI band.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.result_validation import load_accuracy


# Layout of result roots:
#   SAW/FedBuff:  experiment_results/pareto/cifar100-alpha<a>-100users-60Mc-10Buffer/<ALGO>/seed=<s>/cifar100-<ALGO>-<a>-pareto-60-10-<s>-test_acc.txt
#   EqualW/AdaInside: experiment_results/pareto/cifar100_ablation_60_10/<ALGO>/alpha<a>/seed<s>/cifar100-<ALGO>-<a>-pareto-60-10-<s>-test_acc.txt
ROOT = Path(os.environ.get('SAW_RESULTS_ROOT',
            str(Path(__file__).resolve().parents[1] / 'experiment_results/pareto')))
LEGACY_DIR = 'cifar100-alpha{alpha}-100users-60Mc-10Buffer'
ABLATION_DIR = ROOT / 'cifar100_ablation_60_10'

ALPHAS = ['0.01', '0.1', '0.5']
ALPHA_LABELS = {'0.01': r'$\alpha=0.01$', '0.1': r'$\alpha=0.1$', '0.5': r'$\alpha=0.5$'}
SEEDS = [1, 2, 3, 4, 5]
# (method_key, label, is_ablation, color)
METHODS = [
    ('SAW', 'SAW', False, '#1f77b4'),
    ('SAW_EqualW', 'EqualW', True, '#ff7f0e'),
    ('SAW_AdaInside', 'AdaInside', True, '#2ca02c'),
    ('FedBuff', 'FedBuff', False, '#d62728'),
]
SMOOTH_WIN = 10
N_BOOT = 1000
ROUNDS = 500


def _seed_path(method_key: str, alpha: str, seed: int, ablation: bool) -> Path:
    fname = f'cifar100-{method_key}-{alpha}-pareto-60-10-{seed}-test_acc.txt'
    if ablation:
        current = ABLATION_DIR / method_key / f'alpha={alpha}' / f'seed={seed}' / fname
        legacy = ABLATION_DIR / method_key / f'alpha{alpha}' / f'seed{seed}' / fname
        return current if current.is_file() or not legacy.is_file() else legacy
    return ROOT / LEGACY_DIR.format(alpha=alpha) / method_key / f'seed={seed}' / fname


def _load_curves(method_key: str, alpha: str, ablation: bool) -> np.ndarray:
    """Require every seed to have exactly R valid scalar accuracy rows.

    Scalar structure does not establish round provenance.
    """
    curves = []
    for seed in SEEDS:
        path = _seed_path(method_key, alpha, seed, ablation)
        curves.append(load_accuracy(path, expected_rounds=ROUNDS))
    return np.asarray(curves, dtype=float)


def _smooth(x: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    kernel = np.ones(win) / win
    # 'valid' avoids zero-padding artifacts (the end-of-curve cliff).
    valid = np.convolve(x, kernel, mode='valid')
    # Pad both ends with edge values to restore original length.
    pad_left = (x.size - valid.size) // 2
    pad_right = x.size - valid.size - pad_left
    return np.concatenate([
        np.full(pad_left, valid[0]),
        valid,
        np.full(pad_right, valid[-1]),
    ])


def _bootstrap_ci(curves: np.ndarray, n_boot: int = N_BOOT, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, lower, upper) per round from bootstrap resampling of seeds."""
    if curves.shape[0] == 0:
        return np.full(curves.shape[1], np.nan), np.full(curves.shape[1], np.nan), np.full(curves.shape[1], np.nan)
    rng = rng or np.random.default_rng(0)
    n_seeds, R = curves.shape
    boot_means = np.empty((n_boot, R))
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = curves[idx].mean(axis=0)
    lo = np.percentile(boot_means, 2.5, axis=0)
    hi = np.percentile(boot_means, 97.5, axis=0)
    return curves.mean(axis=0), lo, hi


def main() -> None:
    curves = {
        (method_key, alpha): _load_curves(method_key, alpha, ablation)
        for alpha in ALPHAS
        for method_key, _, ablation, _ in METHODS
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    rng = np.random.default_rng(0)

    for ax, alpha in zip(axes, ALPHAS):
        for method_key, label, ablation, color in METHODS:
            raw = curves[(method_key, alpha)]
            smoothed = np.stack([_smooth(raw[i]) for i in range(raw.shape[0])])
            mean, lo, hi = _bootstrap_ci(smoothed, rng=rng)
            x = np.arange(mean.size)
            ax.plot(x, mean, label=f'{label} (n={raw.shape[0]})', color=color, linewidth=1.6)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.set_title(f'({chr(97 + ALPHAS.index(alpha))}) {ALPHA_LABELS[alpha]}')
        ax.set_xlabel('Communication round')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, ROUNDS)

    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].legend(loc='lower right', fontsize=8)

    fig.suptitle(r'Ablation on CIFAR-100, $(M_c, B)=(60, 10)$', y=1.02)
    fig.tight_layout()
    out = Path(os.environ.get('SAW_PLOT_OUTPUT',
                   str(Path.home() / '.local/state/SAW-Implementation/figures/ablation_figure_abc.png')))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

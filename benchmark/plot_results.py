#!/usr/bin/env python3
"""
Plot results from GPU Lanczos benchmark: Naive vs DGKS.

Generates publication-quality figures:
  1. Orthogonality loss over iterations (log scale)
  2. Eigenvalue spectrum comparison with SciPy/ARPACK reference
  3. Eigenvalue relative error vs ARPACK
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator

# Publication-quality defaults
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

# Consistent colors
C_NAIVE = '#d62728'   # red
C_DGKS  = '#1f77b4'   # blue
C_IRLM  = '#9467bd'   # purple
C_REF   = '#2ca02c'   # green


def plot_orthogonality(datadir):
    """Plot orthogonality loss: max|V^T V - I| over Lanczos iterations."""
    data = np.genfromtxt(f'{datadir}/ortho_loss.csv', delimiter=',', names=True)

    iters = data['iteration']
    naive = data['naive']
    dgks = data['dgks']

    mask_naive = naive > 0
    mask_dgks = dgks > 0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(iters[mask_naive], naive[mask_naive], '-o', color=C_NAIVE,
                markersize=4, markeredgecolor='white', markeredgewidth=0.3,
                label='Naive Lanczos (no reorthogonalization)', zorder=3)
    ax.semilogy(iters[mask_dgks], dgks[mask_dgks], '-s', color=C_DGKS,
                markersize=4, markeredgecolor='white', markeredgewidth=0.3,
                label='DGKS Lanczos (ARPACK-style)', zorder=3)

    # Reference lines
    ax.axhline(y=2.2e-16, color='#555555', linestyle='--', linewidth=1, alpha=0.7,
               label=r'Machine $\epsilon$ (double)', zorder=1)
    ax.axhline(y=1.0, color='#999999', linestyle=':', linewidth=1, alpha=0.5,
               label='Complete loss of orthogonality', zorder=1)

    # Shade the region between the two curves to emphasize the gap
    common_mask = mask_naive & mask_dgks
    ax.fill_between(iters[common_mask], dgks[common_mask], naive[common_mask],
                    alpha=0.08, color='purple', zorder=0)

    # Annotate the gap
    mid_idx = len(iters[mask_naive]) * 3 // 4
    mid_iter = iters[mask_naive][mid_idx]
    ax.annotate('', xy=(mid_iter, 3e-1), xytext=(mid_iter, 8e-15),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.5,
                               shrinkA=0, shrinkB=0))
    ax.text(mid_iter + 12, 3e-8, r'$\mathbf{\sim\! 10^{14}\!\times}$',
            fontsize=11, color='#333333', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('Lanczos Iteration')
    ax.set_ylabel(r'Orthogonality Loss: $\max_{i \neq j} |v_i^T v_j|$')
    ax.set_title('Orthogonality Degradation in GPU Lanczos')
    ax.legend(loc='center right', framealpha=0.95)
    ax.set_ylim(1e-17, 1e1)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(f'{datadir}/orthogonality_loss.png', bbox_inches='tight')
    plt.savefig(f'{datadir}/orthogonality_loss.pdf', bbox_inches='tight')
    print(f'Saved: {datadir}/orthogonality_loss.png + .pdf')
    plt.close()


def plot_eigenvalue_comparison(datadir):
    """Compare eigenvalues: spectrum + error vs ARPACK reference."""
    data = np.genfromtxt(f'{datadir}/eigenvalues.csv', delimiter=',', names=True)

    try:
        scipy_eigs = np.loadtxt(f'{datadir}/scipy_eigenvalues.csv')
        has_scipy = True
    except (FileNotFoundError, OSError):
        has_scipy = False

    idx = data['index']
    naive_eigs = data['naive']
    dgks_eigs = data['dgks']
    has_irlm = 'irlm' in data.dtype.names
    irlm_eigs = data['irlm'] if has_irlm else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: eigenvalue spectrum ----
    # Plot order: Naive first (wrong values, clearly separate),
    # then the three correct methods with offset markers so all are visible.
    # ARPACK/DGKS/IRLM overlap perfectly — use different marker sizes and
    # slight horizontal offsets so you can see all three.
    ax = axes[0]

    # Naive: clearly wrong, plotted normally
    ax.plot(idx, naive_eigs, '-o', color=C_NAIVE, markersize=4,
            markeredgecolor='white', markeredgewidth=0.3,
            label='Naive GPU Lanczos', zorder=2)

    # For the three correct methods: use larger → smaller markers so each peeks through
    if has_scipy:
        ax.plot(idx[:len(scipy_eigs)], scipy_eigs[:len(idx)], 's',
                color=C_REF, markersize=8, markeredgecolor=C_REF,
                markeredgewidth=1.5, markerfacecolor='none', linewidth=0,
                label='SciPy ARPACK (reference)', zorder=5)
    ax.plot(idx, dgks_eigs, '-', color=C_DGKS, linewidth=2.5,
            label='DGKS GPU Lanczos', zorder=3)
    if has_irlm:
        ax.plot(idx, irlm_eigs, '-D', color=C_IRLM, markersize=4,
                markeredgecolor='white', markeredgewidth=0.3,
                linewidth=1.5, alpha=0.9,
                label='IRLM GPU Lanczos', zorder=4)

    # Annotate spurious eigenvalues
    n_spurious = np.sum(np.abs(naive_eigs) < 0.1)
    if n_spurious > 1:
        ax.annotate(f'{n_spurious} spurious\nzero copies',
                    xy=(n_spurious // 2, 0.0),
                    xytext=(n_spurious + 1.5, 0.35),
                    fontsize=9, color=C_NAIVE, ha='center',
                    arrowprops=dict(arrowstyle='->', color=C_NAIVE, lw=1.2))

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel(r'$\lambda_i$')
    ax.set_title('Computed Eigenvalue Spectrum')
    ax.legend(loc='upper left', framealpha=0.9)

    # ---- Right: relative error vs ARPACK ----
    ax = axes[1]
    if has_scipy:
        ref = scipy_eigs[:len(idx)]
        k = min(len(ref), len(naive_eigs))
        # Skip λ_0 ≈ 0 (relative error undefined)
        start = 1
        ref_k = ref[start:k]
        naive_err = np.abs(naive_eigs[start:k] - ref_k) / np.abs(ref_k)
        dgks_err = np.abs(dgks_eigs[start:k] - ref_k) / np.abs(ref_k)

        ax.semilogy(idx[start:k], naive_err, '-o', color=C_NAIVE, markersize=4,
                    markeredgecolor='white', markeredgewidth=0.3, label='Naive')
        ax.semilogy(idx[start:k], dgks_err, '-s', color=C_DGKS, markersize=4,
                    markeredgecolor='white', markeredgewidth=0.3, label='DGKS')
        if has_irlm:
            irlm_err = np.abs(irlm_eigs[start:k] - ref_k) / np.abs(ref_k)
            ax.semilogy(idx[start:k], irlm_err, '-D', color=C_IRLM, markersize=4,
                        markeredgecolor='white', markeredgewidth=0.3, label='IRLM')

        # Reference lines
        ax.axhline(y=2.2e-16, color='#555555', linestyle='--', linewidth=1,
                   alpha=0.6, label=r'Machine $\epsilon$')

        ax.set_ylabel('Relative Error vs ARPACK')
        ax.set_ylim(bottom=1e-17, top=1e1)
    else:
        diff = np.abs(naive_eigs - dgks_eigs)
        ref_safe = np.where(np.abs(dgks_eigs) < 1e-15, 1e-15, dgks_eigs)
        rel_diff = diff / np.abs(ref_safe)
        ax.semilogy(idx, rel_diff, '-o', color='black', markersize=4)
        ax.set_ylabel('Relative Difference (Naive vs DGKS)')

    ax.set_xlabel('Eigenvalue Index')
    ax.set_title('Eigenvalue Accuracy vs ARPACK Reference')
    ax.legend(loc='center left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{datadir}/eigenvalue_comparison.png', bbox_inches='tight')
    plt.savefig(f'{datadir}/eigenvalue_comparison.pdf', bbox_inches='tight')
    print(f'Saved: {datadir}/eigenvalue_comparison.png + .pdf')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data', help='Directory with CSV results')
    args = parser.parse_args()

    plot_orthogonality(args.datadir)
    plot_eigenvalue_comparison(args.datadir)
    print('Done!')

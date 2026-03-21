#!/usr/bin/env python3
"""
Compute reference eigenvalues using SciPy's ARPACK wrapper (eigsh).
This provides the ground truth for evaluating GPU Lanczos accuracy.

Loads the exact Laplacian matrix exported by the CUDA benchmark (Matrix Market
format) so the comparison is apples-to-apples — same matrix, same eigenvalues.
"""

import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eigs', type=int, default=20)
    parser.add_argument('--outdir', default='data')
    args = parser.parse_args()

    mtx_file = f"{args.outdir}/laplacian.mtx"
    print(f"Loading Laplacian from {mtx_file}")
    L = mmread(mtx_file).tocsr()
    print(f"  Shape: {L.shape}, nnz: {L.nnz}")

    print(f"\nComputing {args.eigs} smallest eigenvalues with SciPy ARPACK...")
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(L, k=args.eigs, which='SM', tol=1e-10)
    t1 = time.time()

    # Sort by ascending eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]

    print(f"  Time: {(t1 - t0) * 1000:.1f} ms")
    print(f"  Eigenvalues: {eigenvalues[:10]}")
    if len(eigenvalues) > 10:
        print(f"  ... ({len(eigenvalues)} total)")

    # Save reference eigenvalues
    outfile = f"{args.outdir}/scipy_eigenvalues.csv"
    np.savetxt(outfile, eigenvalues, fmt='%.15e')
    print(f"\nSaved reference eigenvalues to {outfile}")

    # Compare with GPU results if available
    try:
        gpu_data = np.genfromtxt(f'{args.outdir}/eigenvalues.csv',
                                  delimiter=',', names=True)
        naive_eigs = gpu_data['naive']
        dgks_eigs = gpu_data['dgks']
        has_irlm = 'irlm' in gpu_data.dtype.names
        irlm_eigs = gpu_data['irlm'] if has_irlm else None

        k = min(len(eigenvalues), len(naive_eigs))
        ref = eigenvalues[:k]

        print(f"\n{'='*72}")
        print("ACCURACY COMPARISON vs SciPy ARPACK (ground truth)")
        print(f"{'='*72}")
        hdr = f"{'Index':>6} {'ARPACK':>14} {'Naive err':>14} {'DGKS err':>14}"
        if has_irlm: hdr += f" {'IRLM err':>14}"
        print(hdr)
        print(f"{'-'*6} {'-'*14} {'-'*14} {'-'*14}" + (f" {'-'*14}" if has_irlm else ""))

        for i in range(k):
            ref_val = ref[i]
            denom = max(abs(ref_val), 1e-15)
            naive_err = abs(naive_eigs[i] - ref_val) / denom
            dgks_err = abs(dgks_eigs[i] - ref_val) / denom
            line = f"{i:>6d} {ref_val:>14.8f} {naive_err:>14.2e} {dgks_err:>14.2e}"
            if has_irlm:
                irlm_err = abs(irlm_eigs[i] - ref_val) / denom
                line += f" {irlm_err:>14.2e}"
            print(line)

        naive_max_err = max(abs(naive_eigs[:k] - ref) / np.maximum(abs(ref), 1e-15))
        dgks_max_err = max(abs(dgks_eigs[:k] - ref) / np.maximum(abs(ref), 1e-15))
        msg = f"\nMax relative error:  Naive={naive_max_err:.2e}  DGKS={dgks_max_err:.2e}"
        if has_irlm:
            irlm_max_err = max(abs(irlm_eigs[:k] - ref) / np.maximum(abs(ref), 1e-15))
            msg += f"  IRLM={irlm_max_err:.2e}"
        print(msg)

    except (FileNotFoundError, OSError):
        print("\n(Run the GPU benchmark first to compare: make run)")


if __name__ == '__main__':
    main()

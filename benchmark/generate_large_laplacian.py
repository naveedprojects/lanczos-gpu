#!/usr/bin/env python3
"""Generate a large KNN graph Laplacian in Matrix Market format.

Uses scipy.spatial.cKDTree for fast O(n log n) KNN and vectorized
numpy operations for the sparse matrix construction.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--bw', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--outdir', default='data')
    args = parser.parse_args()

    n = args.n
    print(f"Generating n={n:,}, dim={args.dim}, k={args.k}, bw={args.bw}")
    t0 = time.time()

    rng = np.random.RandomState(args.seed)
    points = rng.random((n, args.dim))

    print(f"  Building KD-tree... ({time.time()-t0:.0f}s)")
    tree = cKDTree(points)
    del points

    print(f"  Querying KNN... ({time.time()-t0:.0f}s)")
    distances, indices = tree.query(tree.data, k=args.k + 1)
    del tree

    print(f"  Building sparse adjacency (vectorized)... ({time.time()-t0:.0f}s)")
    # Vectorized construction — no Python for-loop
    # Skip self-neighbors (column 0)
    neighbor_dists = distances[:, 1:]   # (n, k)
    neighbor_idxs = indices[:, 1:]      # (n, k)
    del distances, indices

    weights = np.exp(-neighbor_dists**2 / (4.0 * args.bw**2))  # (n, k)
    del neighbor_dists

    rows = np.repeat(np.arange(n), args.k)             # each row repeated k times
    cols = neighbor_idxs.ravel()                         # flatten
    vals = weights.ravel()
    del neighbor_idxs, weights

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    del rows, cols, vals

    print(f"  Symmetrizing... ({time.time()-t0:.0f}s)")
    A_sym = A.maximum(A.T)
    del A

    print(f"  Building Laplacian L = D - A... ({time.time()-t0:.0f}s)")
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (np.arange(n), np.arange(n))), shape=(n, n))
    L = D - A_sym
    del D, A_sym

    t1 = time.time()
    outfile = f"{args.outdir}/laplacian.mtx"
    print(f"  Writing {outfile}... ({time.time()-t0:.0f}s)")
    mmwrite(outfile, L)

    t2 = time.time()
    print(f"Done: n={n:,}, nnz={L.nnz:,}, build={t1-t0:.1f}s, write={t2-t1:.1f}s")
    print(f"Saved: {outfile}")


if __name__ == '__main__':
    main()

/**
 * tridiag.cuh — CPU-side tridiagonal matrix operations.
 *
 * All functions here operate on small host-side arrays (size ncv).
 * No GPU memory is touched.
 *
 * Contents:
 *   givens()                — Stable Givens rotation
 *   apply_shifts_tridiag()  — Implicit QR bulge chase for IRLM
 *   solve_tridiag()         — LAPACK dstev wrapper
 */

#pragma once

#include <cmath>
#include <cstring>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// LAPACK symmetric tridiagonal eigensolver
extern "C" void dstev_(const char *jobz, const int *n, double *d, double *e,
                       double *z, const int *ldz, double *work, int *info);

// ============================================================
// Givens rotation: compute (c, s) such that
//   [ c  s] [a]   [r]
//   [-s  c] [b] = [0]
//
// Uses the stable formulation from Golub & Van Loan (Algorithm 5.1.3)
// that avoids overflow regardless of the magnitudes of a and b.
// ============================================================
inline void givens(double a, double b, double &c, double &s) {
    if (fabs(b) < DBL_TRUE_MIN) {
        // b is essentially zero (below smallest subnormal)
        c = 1.0;
        s = 0.0;
    } else if (fabs(b) > fabs(a)) {
        double tau = -a / b;
        s = 1.0 / sqrt(1.0 + tau * tau);
        c = s * tau;
    } else {
        double tau = -b / a;
        c = 1.0 / sqrt(1.0 + tau * tau);
        s = c * tau;
    }
}

// ============================================================
// Apply np implicit QR shifts to a symmetric tridiagonal matrix.
//
// The tridiagonal T is stored as:
//   d[0..m-1]  = diagonal
//   e[0..m-2]  = subdiagonal, where e[i] = T(i+1, i)
//
// For each shift sigma, performs one QR step:
//   Factor (T - sigma*I) = QR via Givens rotations
//   T <- Q^T T Q  (bulge chase)
//   Accumulate Q into Qmat (m x m, column-major)
//
// On exit: d, e hold the updated tridiagonal; Qmat holds the
// product Q_1 Q_2 ... Q_np of all rotation matrices.
//
// This implements the same algorithm as ARPACK's dsapps.f,
// specialized for exact shifts on a symmetric tridiagonal.
// ============================================================
inline void apply_shifts_tridiag(double *d, double *e, int m,
                                 const double *shifts, int np,
                                 double *Qmat) {
    // Initialize Q = I
    memset(Qmat, 0, (size_t)m * m * sizeof(double));
    for (int i = 0; i < m; i++)
        Qmat[i + i * m] = 1.0;

    for (int is = 0; is < np; is++) {
        double sigma = shifts[is];

        // First Givens rotation: zero e[0] in column 0 of (T - sigma*I)
        double c, s;
        givens(d[0] - sigma, e[0], c, s);

        // Apply G(0,1)^T * T * G(0,1)
        double d0 = d[0], d1 = d[1], e0 = e[0];
        double e1 = (m > 2) ? e[1] : 0.0;

        d[0] = c * c * d0 - 2.0 * c * s * e0 + s * s * d1;
        d[1] = s * s * d0 + 2.0 * c * s * e0 + c * c * d1;
        e[0] = c * s * (d0 - d1) + (c * c - s * s) * e0;

        double bulge = -s * e1;
        if (m > 2) e[1] = c * e1;

        // Accumulate into Q
        for (int i = 0; i < m; i++) {
            double q0 = Qmat[i + 0 * m], q1 = Qmat[i + 1 * m];
            Qmat[i + 0 * m] =  c * q0 - s * q1;
            Qmat[i + 1 * m] =  s * q0 + c * q1;
        }

        // Chase the bulge down the tridiagonal
        for (int k = 1; k < m - 1; k++) {
            givens(e[k - 1], bulge, c, s);

            double dk = d[k], dk1 = d[k + 1], ek = e[k];
            double ek1 = (k + 2 < m) ? e[k + 1] : 0.0;

            e[k - 1] = c * e[k - 1] - s * bulge;
            d[k]     = c * c * dk - 2.0 * c * s * ek + s * s * dk1;
            d[k + 1] = s * s * dk + 2.0 * c * s * ek + c * c * dk1;
            e[k]     = c * s * (dk - dk1) + (c * c - s * s) * ek;

            if (k + 2 < m) {
                bulge    = -s * ek1;
                e[k + 1] =  c * ek1;
            }

            for (int i = 0; i < m; i++) {
                double qk = Qmat[i + k * m], qk1 = Qmat[i + (k + 1) * m];
                Qmat[i + k * m]       =  c * qk - s * qk1;
                Qmat[i + (k + 1) * m] =  s * qk + c * qk1;
            }
        }
    }
}

// ============================================================
// Solve the symmetric tridiagonal eigenproblem.
//
// Wraps LAPACK's dstev: compute eigenvalues and eigenvectors
// of the m x m tridiagonal with diagonal alpha[0..m-1] and
// subdiagonal beta[1..m-1] (beta[0] unused, beta[i] = T(i,i-1)).
//
// On exit:
//   eig_d[0..m-1]             = eigenvalues in ascending order
//   eig_z[0..m*m-1]           = eigenvectors as columns (col-major)
//   return value               = LAPACK info (0 = success)
// ============================================================
inline int solve_tridiag(const double *alpha, const double *beta, int m,
                         double *eig_d, double *eig_z) {
    double *eig_e = (double *)malloc(m * sizeof(double));
    double *work  = (double *)malloc(2 * m * sizeof(double));

    memcpy(eig_d, alpha, m * sizeof(double));
    for (int i = 0; i < m - 1; i++)
        eig_e[i] = beta[i + 1];

    int info;
    char jobz = 'V';
    dstev_(&jobz, &m, eig_d, eig_e, eig_z, &m, work, &info);

    ::free(eig_e);
    ::free(work);

    if (info != 0) {
        fprintf(stderr, "LAPACK dstev failed: info = %d\n", info);
    }
    return info;
}

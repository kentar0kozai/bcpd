// Copyright (c) 2018-2019 Osamu Hirose
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../../../register/bcpd-util.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#include "../../../../base/gaussprod.h"
#include "../../../../base/gramdecomp.h"
#include "../../../../base/kdtree.h"
#include "../../../../base/kernel.h"
#include "../../../../base/lapack.h"
#include "../../../../base/misc.h"

#define SQ(x) ((x) * (x))
double (*kernel[4])(const double *, const double *, int, double) = {gauss, imquad, rational, laplace};

void interpolate(double *T, const double *Y, const int N, const double *x, const double *y, const double *w, const double *s, const double *R,
                 const double *t, const double *r, const pwsz sz, const pwpm pm) {

    int d, i, j, k, m, n;
    int D = sz.D, K = sz.K, M = sz.M;
    int *wi;
    double *u, *ix, *A, *B, *E, *G, *L, *Q, *W, *wd;
    char uplo = 'U';
    int *U;
    int info;
    double bet = pm.bet, lmd = pm.lmd;
    double val, cc = pm.lmd * SQ(*r / (*s));
    int si = sizeof(int), sd = sizeof(double);

    /* allocation */
    W = calloc(D * M, sd);
    ix = calloc(D * M, sd);
    E = calloc(D * M, sd);
    u = calloc(D * N, sd);
    /* switch: non-rigid and rigid */
    if (lmd >= 1e8) {
        for (d = 0; d < D; d++)
            for (n = 0; n < N; n++) {
                { u[d + D * n] = Y[d + D * n]; }
            }
        goto skip;
    }
    /* non-rigid */
    cc = lmd * SQ(*r / (*s));
    for (d = 0; d < D; d++)
        for (m = 0; m < M; m++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[i + D * d] * (x[i + D * m] - t[i]);
            }
            ix[d + D * m] = val / (*s);
        }
    for (d = 0; d < D; d++)
        for (m = 0; m < M; m++)
            E[d + D * m] = W[m + M * d] = ix[d + D * m] - y[d + D * m];
    if (K) { /* nystrom */
        /* coefficient: W */
        A = calloc(K * K, sd);
        L = calloc(K, sd);
        wd = calloc(K * (K + 11), sd);
        B = calloc(K * D, sd);
        U = calloc(N, si);
        Q = calloc(M * K, sd);
        wi = calloc(M, si);
        gramdecomp(Q, L, wd, wi, y, D, M, K, bet, kernel[pm.G]);
#pragma omp parallel for private(j) private(m) private(val)
        for (i = 0; i < K; i++)
            for (j = i; j < K; j++) {
                val = (i == j ? cc / L[i] : 0);
                for (m = 0; m < M; m++) {
                    val += w[m] * Q[m + M * i] * Q[m + M * j];
                }
                A[i + K * j] = A[j + K * i] = val;
            }
        for (d = 0; d < D; d++)
            for (m = 0; m < M; m++)
                E[d + D * m] *= w[m];
        for (k = 0; k < K; k++)
            for (d = 0; d < D; d++) {
                val = 0;
                for (m = 0; m < M; m++) {
                    val += Q[m + M * k] * E[d + D * m];
                }
                B[k + K * d] = val;
            }
        dposv_(&uplo, &K, &D, A, &K, B, &K, &info);
        assert(!info);
        for (k = 0; k < K; k++)
            for (m = 0; m < M; m++)
                Q[m + M * k] *= w[m];
        for (m = 0; m < M; m++)
            for (d = 0; d < D; d++) {
                val = E[d + D * m];
                for (k = 0; k < K; k++) {
                    val -= Q[m + M * k] * B[k + K * d];
                }
                W[m + M * d] = val / cc;
            }
        /* interpolation */
        randperm(U, N);
        for (i = 0; i < K; i++)
            for (j = i; j < K; j++)
                A[i + K * j] = A[j + K * i] = kernel[pm.G](Y + D * U[i], Y + D * U[j], D, bet) + (i == j ? 1e-9 : 0);
        for (k = 0; k < K; k++)
            for (d = 0; d < D; d++) {
                val = 0;
                for (m = 0; m < M; m++) {
                    val += kernel[pm.G](Y + D * U[k], y + D * m, D, bet) * W[m + M * d];
                }
                B[k + K * d] = val;
            }
        dposv_(&uplo, &K, &D, A, &K, B, &K, &info);
        assert(!info);
        for (d = 0; d < D; d++)
            for (n = 0; n < N; n++) {
                val = 0;
                for (k = 0; k < K; k++) {
                    val += kernel[pm.G](Y + D * n, Y + D * U[k], D, bet) * B[k + K * d];
                }
                u[d + D * n] = val + Y[d + D * n];
            }
        free(A);
        free(Q);
        free(wd);
        free(U);
        free(B);
        free(L);
        free(wi);
    } else { /* direct */
        /* coefficient: W */
        G = calloc(M * M, sd);
#pragma omp parallel for private(j)
        for (i = 0; i < M; i++)
            for (j = i; j < M; j++)
                G[i + M * j] = G[j + M * i] = kernel[pm.G](y + D * i, y + D * j, D, bet) + (i == j ? cc / w[i] : 0);
        dposv_(&uplo, &M, &D, G, &M, W, &M, &info);
        assert(!info);
/* interpolation */
#pragma omp parallel for private(d) private(m) private(val)
        for (n = 0; n < N; n++)
            for (d = 0; d < D; d++) {
                val = 0;
                for (m = 0; m < M; m++) {
                    val += kernel[pm.G](Y + D * n, y + D * m, D, bet) * W[m + M * d];
                }
                u[d + D * n] = val + Y[d + D * n];
            }
        free(G);
    }
skip:
    for (d = 0; d < D; d++)
        for (n = 0; n < N; n++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[d + D * i] * u[i + D * n];
            }
            T[d + D * n] = (*s) * val + t[d];
        }
    free(W);
    free(ix);
    free(E);
    free(u);
}

/* y: downsampled data */
void interpolate_1nn(double *T, const double *Y, const int N, const double *v, const double *y, const double *s, const double *R, const double *t,
                     const pwsz sz, const pwpm pm) {

    int d, i, n, *m, *bi, *Ty;
    int D = sz.D, M = sz.M;
    double *e, *U, *bd;
    double val;
    int si = sizeof(int), sd = sizeof(double);

    /* allocation */
    m = calloc(N, si);
    bi = calloc(6 * M, si);
    e = calloc(N, sd);
    bd = calloc(2 * M, sd);
    U = calloc(D * N, sd);
    Ty = calloc(3 * M + 1, si);
    /* kdtree */
    kdtree(Ty, bi, bd, y, D, M);
/* 1nn */
#pragma omp parallel for
    for (n = 0; n < N; n++)
        nnsearch(m + n, e + n, Y + D * n, y, Ty, D, M);
    for (n = 0; n < N; n++)
        for (d = 0; d < D; d++)
            U[d + D * n] = Y[d + D * n] + v[d + D * m[n]];
    for (d = 0; d < D; d++)
        for (n = 0; n < N; n++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[d + D * i] * U[i + D * n];
            }
            T[d + D * n] = (*s) * val + t[d];
        }
    /* free */
    free(m);
    free(e);
    free(U);
    free(bi);
    free(bd);
    free(Ty);
}

void interpolate_geok(double *T, const double *Y, const int N, const double *x, const double *y, const double *w, const double *s, const double *R,
                      const double *t, const double *r, const double *LQ, const int *U, const pwsz sz, const pwpm pm) {

    int d, i, j, k, m, n;
    int D = sz.D, K = sz.K, M = sz.M;
    double *u, *ix, *A, *B, *C, *S, *E, *W;
    char uplo = 'U';
    int info;
    const double lmd = pm.lmd;
    double val, cc;
    int sd = sizeof(double);
    const double *L = LQ, *Q = LQ + K;
    double *Qy;

    assert(K);

    /* allocation */
    W = calloc(D * M, sd);
    ix = calloc(D * M, sd);
    E = calloc(D * M, sd);
    u = calloc(D * N, sd);
    A = calloc(K * K, sd);
    Qy = calloc(K * M, sd);
    B = calloc(K * D, sd);
    C = calloc(M * K, sd);
    S = calloc(K * K, sd);

    for (k = 0; k < K; k++)
        for (m = 0; m < M; m++)
            Qy[m + M * k] = Q[U[m] + N * k];
    /* switch: non-rigid and rigid */
    if (lmd >= 1e8) {
        for (d = 0; d < D; d++)
            for (n = 0; n < N; n++) {
                { u[d + D * n] = Y[d + D * n]; }
            }
        goto skip;
    }

    cc = lmd * SQ(*r / (*s));
    for (d = 0; d < D; d++)
        for (m = 0; m < M; m++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[i + D * d] * (x[i + D * m] - t[i]);
            }
            ix[d + D * m] = val / (*s);
        }
    for (m = 0; m < M; m++)
        for (d = 0; d < D; d++)
            E[m + M * d] = ix[d + D * m] - y[d + D * m];
    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++)
            C[m + M * k] = w[m] * Qy[m + M * k];
    for (k = 0; k < K; k++)
        for (d = 0; d < D; d++) {
            B[k + K * d] = 0;
            for (m = 0; m < M; m++)
                B[k + K * d] += C[m + M * k] * E[m + M * d];
        }
#pragma omp parallel for private(j) private(m) private(val)
    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++) {
            val = 0;
            for (m = 0; m < M; m++) {
                val += Qy[m + M * i] * C[m + M * j];
            }
            A[i + K * j] = val;
        }
    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++)
            S[i + K * j] = A[i + K * j];
    for (k = 0; k < K; k++)
        A[k + K * k] += cc / L[k];
    dpotrf_(&uplo, &K, A, &K, &info);
    assert(info == 0);
    dpotrs_(&uplo, &K, &D, A, &K, B, &K, &info);
    assert(info == 0);
    dpotrs_(&uplo, &K, &K, A, &K, S, &K, &info);
    assert(info == 0);
    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++)
            A[i + K * j] = L[i] * ((i == j ? 1 : 0) - S[j + K * i]);
    for (m = 0; m < M; m++)
        for (d = 0; d < D; d++)
            W[d + D * m] = w[m] * E[m + M * d];
    for (m = 0; m < M; m++)
        for (d = 0; d < D; d++)
            for (k = 0; k < K; k++)
                W[d + D * m] -= C[m + M * k] * B[k + K * d];
    for (m = 0; m < M; m++)
        for (d = 0; d < D; d++)
            W[d + D * m] /= cc;
    /* u */
    for (k = 0; k < K; k++)
        for (d = 0; d < D; d++) {
            val = 0;
            for (m = 0; m < M; m++) {
                val += Qy[m + M * k] * W[d + D * m];
            }
            B[k + K * d] = val * L[k];
        }
    for (d = 0; d < D; d++)
        for (n = 0; n < N; n++) {
            val = 0;
            for (k = 0; k < K; k++) {
                val += Q[n + N * k] * B[k + K * d];
            }
            u[d + D * n] = val + Y[d + D * n];
        }

skip:
    for (d = 0; d < D; d++)
        for (n = 0; n < N; n++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[d + D * i] * u[i + D * n];
            }
            T[d + D * n] = (*s) * val + t[d];
        }

    free(A);
    free(Qy);
    free(B);
    free(ix);
    free(E);
    free(u);
    free(W);
}

void interpolate_x(double *x, const double *y, const double *X, int D, int M, int N, const double r, pwpm pm) {
    int d, k, m, K = 30;
    double e, val;
    double *p, *P;
    int *T, *q, *Q;
    int me = -1;
    int si = sizeof(int), sd = sizeof(double);

    K = K < M ? K : M;

    P = calloc(M * (K + 1), sd);
    T = kdtree_build(X, D, N);
    Q = calloc(M * (K + 1), si);

    //#pragma omp parallel for private (q) private (p) private (d) private (e) private (k)
    for (m = 0; m < M; m++) {
        q = Q + (K + 1) * m;
        p = P + (K + 1) * m;
        knnsearch(q, K, 2 * r, y + D * m, me, X, T, D, N);
        if (*q) {
            *p = 0;
            for (k = 1; k <= *q; k++) {
                p[k] = gauss(y + D * m, X + D * q[k], D, r);
                *p += p[k];
            }
        } else {
            *p = 1.0;
            p[1] = 1.0;
            *q = 1;
            nnsearch(q + 1, &e, y + D * m, X, T, D, N);
        }
        for (d = 0; d < D; d++) {
            val = 0;
            for (k = 1; k <= *q; k++) {
                val += p[k] * X[d + D * q[k]];
            }
            x[d + D * m] = val / (*p);
        }
    }
    free(P);
    free(Q);
    free(T);

    return;
}

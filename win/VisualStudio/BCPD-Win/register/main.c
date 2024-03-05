// Copyright (c) 2018-2020 Osamu Hirose
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

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#include "../../../../base/geokdecomp.h"
#include "../../../../base/kdtree.h"
#include "../../../../base/kernel.h"
#include "../../../../base/misc.h"
#include "../../../../base/sampling.h"
#include "../../../../base/sgraph.h"
#include "../../../../base/util.h"
#include "../../../../register/bcpd.h"
#include "../../../../register/info.h"
#include "../../../../register/norm.h"
#include "getopt.h"

#define SQ(x) ((x) * (x))
#define M_PI 3.14159265358979323846 // pi

void init_genrand64(unsigned long s);
enum transpose { ASIS = 0,
    TRANSPOSE = 1 };

void save_variable(const char* prefix, const char* suffix, const double* var,
    int D, int J, char* fmt, int trans)
{
    int d, j;
    char fn[256];
    double** buf;
    strcpy(fn, prefix);
    strcat(fn, suffix);
    if (trans == TRANSPOSE) {
        buf = calloc2d(J, D);
        for (j = 0; j < J; j++)
            for (d = 0; d < D; d++)
                buf[j][d] = var[d + D * j];
        write2d(fn, (const double**)buf, J, D, fmt, "NA");
        free2d(buf, J);
    } else {
        buf = calloc2d(D, J);
        for (j = 0; j < J; j++)
            for (d = 0; d < D; d++)
                buf[d][j] = var[d + D * j];
        write2d(fn, (const double**)buf, D, J, fmt, "NA");
        free2d(buf, D);
    }

    return;
}

void save_corresp(const char* prefix, const double* X, const double* y,
    const double* a, const double* sgm, const double s,
    const double r, pwsz sz, pwpm pm)
{
    int i, m, n, D, M, N;
    int *T, *l, *bi;
    double* bd;
    double *p, c, val;
    char fnP[256], fnc[256], fne[256];
    int S[MAXTREEDEPTH];
    int top, ct;
    double omg, dlt, vol, rad;
    int si = sizeof(int), sd = sizeof(double);
    FILE *fpP = NULL, *fpc = NULL, *fpe = NULL;
    int db = pm.opt & PW_OPT_DBIAS;
    double max, min;
    int mmax;

    D = sz.D;
    M = sz.M;
    N = sz.N;
    omg = pm.omg;
    dlt = pm.dlt;
    rad = dlt * r;
    strcpy(fnP, prefix);
    strcat(fnP, "P.txt");
    if (pm.opt & PW_OPT_SAVEP) {
        fpP = fopen(fnP, "w");
        fprintf(fpP, "[n]\t[m]\t[probability]\n");
    }
    strcpy(fne, prefix);
    strcat(fne, "e.txt");
    if (pm.opt & PW_OPT_SAVEE) {
        fpe = fopen(fne, "w");
        fprintf(fpe, "[n]\t[m]\t[probability]\n");
    }
    strcpy(fnc, prefix);
    strcat(fnc, "c.txt");
    if (pm.opt & PW_OPT_SAVEC) {
        fpc = fopen(fnc, "w");
        fprintf(fpc, "[n]\t[1/0]\n");
    }

    // T = calloc(3 * M + 1, si); bi = calloc(6 * M, si); bd = calloc(2 * M, sd);
    // p = calloc(M, sd); l = calloc(M, si);

    T = calloc((size_t)3 * M + 1, si);
    bi = calloc((size_t)6 * M, si);
    bd = calloc((size_t)2 * M, sd);
    p = calloc((size_t)M, sd);
    l = calloc((size_t)M, si);
    if (l == NULL) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return 1;
    }

    kdtree(T, bi, bd, y, D, M);
    vol = volume(X, D, N);
    c = (pow(2.0 * M_PI * SQ(r), 0.5 * D) * omg) / (vol * (1 - omg));
    for (n = 0; n < N; n++) {
        /* compute P, c, e */
        val = c;
        top = ct = 0;
        do {
            eballsearch_next(&m, S, &top, X + (size_t)D * n, rad, y, T, D, M);
            if (m >= 0)
                l[ct++] = m;
        } while (top);
        if (!ct) {
            nnsearch(&m, &min, X + (size_t)D * n, y, T, D, M);
            l[ct++] = m;
        }
        for (i = 0; i < ct; i++) {
            m = l[i];
            p[i] = a[m] * gauss(y + (size_t)D * m, X + (size_t)D * n, D, r) * (db ? exp(-0.5 * D * SQ(sgm[m] * s / r)) : 1.0);
            val += p[i];
        }
        for (i = 0; i < ct; i++) {
            m = l[i];
            p[i] /= val;
        }
        max = c / val;
        mmax = 0;
        for (i = 0; i < ct; i++)
            if (p[i] > max) {
                max = p[i];
                mmax = l[i] + 1;
            }
        /* print P, c, e */
        if (fpP) {
            for (i = 0; i < ct; i++)
                if (p[i] > 1.0f / M) {
                    m = l[i];
                    fprintf(fpP, "%d\t%d\t%lf\n", n + 1, m + 1, p[i]);
                }
        }
        if (fpe) {
            fprintf(fpe, "%d\t%d\t%lf\n", n + 1, mmax ? mmax : l[0],
                mmax ? max : p[0]);
        }
        if (fpc) {
            fprintf(fpc, "%d\t%d\n", n + 1, mmax ? 1 : 0);
        }
    }
    if (fpP) {
        fclose(fpP);
    }
    free(l);
    free(bd);
    if (fpe) {
        fclose(fpc);
    }
    free(p);
    free(bi);
    if (fpc) {
        fclose(fpe);
    }
    free(T);
    return;
}

int save_optpath(const char* file, const double* sy, const double* X, pwsz sz,
    pwpm pm, int lp)
{
    int N = sz.N, M = sz.M, D = sz.D;
    int si = sizeof(int), sd = sizeof(double);
    FILE* fp = fopen(file, "wb");
    if (!fp) {
        printf("Can't open: %s\n", file);
        exit(EXIT_FAILURE);
    }
    fwrite(&N, si, 1, fp);
    fwrite(&D, si, 1, fp);
    fwrite(&M, si, 1, fp);
    fwrite(&lp, si, 1, fp);
    fwrite(sy, sd, (size_t)lp * D * M, fp);
    fwrite(X, sd, (size_t)D * N, fp);
    if (strlen(pm.fn[FACE_Y])) {
        double** b;
        int nl, nc, l, c;
        char mode;
        int* L = NULL;
        b = read2d(&nl, &nc, &mode, pm.fn[FACE_Y], "NA");
        assert(nc == 3 || nc == 2);
        L = calloc((size_t)nc * nl, si);
        if (L == NULL) {
            fprintf(stderr, "Failed to allocate memory.\n");
            return 1;
        }
        for (l = 0; l < nl; l++)
            for (c = 0; c < nc; c++) {
                L[c + nc * l] = (int)b[l][c];
            }
        fwrite(&nl, si, 1, fp);
        fwrite(&nc, si, 1, fp);
        fwrite(L, si, (size_t)nc * nl, fp);
        free(L);
        free(b);
    }
    fclose(fp);

    return 0;
}

void scan_kernel(pwpm* pm, const char* arg)
{
    char* p;
    if ('g' != tolower(*arg)) {
        pm->G = atoi(arg);
        return;
    }
    p = strchr(arg, ',');
    if (!p) {
        printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (strstr(arg, "txt"))
        sscanf(p + 1, "%lf,%s", &(pm->tau), pm->fn[FACE_Y]);
    else
        sscanf(p + 1, "%lf,%d,%lf", &(pm->tau), &(pm->nnk), &(pm->nnr));
    if (pm->tau < 0 || pm->tau > 1) {
        printf(
            "ERROR: the 2nd argument of -G (tau) must be in range [0,1]. Abort.\n");
        exit(EXIT_FAILURE);
    }
}

void scan_dwpm(int* dwn, double* dwr, const char* arg)
{
    char c;
    int n, m;
    double r;
    m = sscanf(arg, "%c,%d,%lf", &c, &n, &r);
    if (m != 3)
        goto err01;
    if (n <= 0)
        goto err03;
    if (r < 0)
        goto err04;
    if (isupper(c)) {
        r *= -1.0f;
    }
    c = tolower(c);
    if (c != 'x' && c != 'y' && c != 'b')
        goto err02;
    if (r < 0 && -r < 1e-2)
        r = -1e-2;
    switch (c) {
    case 'x':
        dwn[TARGET] = n;
        dwr[TARGET] = r;
        break;
    case 'y':
        dwn[SOURCE] = n;
        dwr[SOURCE] = r;
        break;
    case 'b':
        dwn[TARGET] = n;
        dwr[TARGET] = r;
        dwn[SOURCE] = n;
        dwr[SOURCE] = r;
        break;
    }
    return;
err01:
    printf("ERROR: The argument of '-D' must be 'char,int,real'. \n");
    exit(EXIT_FAILURE);
err02:
    printf("ERROR: The 1st argument of '-D' must be one of [x,y,b,X,Y,B]. \n");
    exit(EXIT_FAILURE);
err03:
    printf("ERROR: The 2nd argument of '-D' must be positive.     \n");
    exit(EXIT_FAILURE);
err04:
    printf("ERROR: The 3rd argument of '-D' must be positive or 0.\n");
    exit(EXIT_FAILURE);
}

void check_prms(const pwpm pm, const pwsz sz)
{
    int M = sz.M, N = sz.N, M0 = pm.dwn[SOURCE], N0 = pm.dwn[TARGET];
    M = M0 ? M0 : M;
    N = N0 ? N0 : N;
    if (pm.nlp <= 0) {
        printf("ERROR: -n: Argument must be a positive integer. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.llp <= 0) {
        printf("ERROR: -N: Argument must be a positive integer. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.omg < 0) {
        printf("ERROR: -w: Argument must be in range [0,1]. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.omg >= 1) {
        printf("ERROR: -w: Argument must be in range [0,1]. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.lmd <= 0) {
        printf("ERROR: -l: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.kpa <= 0) {
        printf("ERROR: -k: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.dlt <= 0) {
        printf("ERROR: -d: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.lim <= 0) {
        printf("ERROR: -e: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.btn <= 0) {
        printf("ERROR: -f: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.cnv <= 0) {
        printf("ERROR: -c: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.rns < 0) {
        printf("ERROR: -r: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.K < 0) {
        printf("ERROR: -K: Argument must be a positive integer. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.K > M) {
        printf("ERROR: -K: Argument must be less than M. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.J < 0) {
        printf("ERROR: -J: Argument must be a positive integer. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.J > M + N) {
        printf("ERROR: -J: Argument must be less than M+N. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.G > 3) {
        printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.G < 0) {
        printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.bet <= 0) {
        printf("ERROR: -b: Argument must be positive. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.eps < 0) {
        printf("ERROR: -z: Argument must be in range [0,1]. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (pm.eps > 1) {
        printf("ERROR: -z: Argument must be in range [0,1]. Abort.\n");
        exit(EXIT_FAILURE);
    }
    if (!strchr("exyn", pm.nrm)) {
        printf(
            "\n  ERROR: -u: Argument must be one of 'e', 'x', 'y' and 'n'. "
            "Abort.\n\n");
        exit(EXIT_FAILURE);
    }
}

void pw_getopt(pwpm* pm, int argc, char** argv)
{
    int opt;
    strcpy(pm->fn[TARGET], "X.txt");
    pm->omg = 0.0;
    pm->cnv = 1e-4;
    pm->K = 0;
    pm->opt = 0.0;
    pm->btn = 0.20;
    pm->bet = 2.0;
    strcpy(pm->fn[SOURCE], "Y.txt");
    pm->lmd = 2.0;
    pm->nlp = 500;
    pm->J = 0;
    pm->dlt = 7.0;
    pm->lim = 0.15;
    pm->eps = 1e-3;
    strcpy(pm->fn[OUTPUT], "output_");
    pm->rns = 0;
    pm->llp = 30;
    pm->G = 0;
    pm->gma = 1.0;
    pm->kpa = ZERO;
    pm->nrm = 'e';
    strcpy(pm->fn[FACE_Y], "");
    pm->nnk = 0;
    pm->nnr = 0;
    strcpy(pm->fn[FUNC_Y], "");
    strcpy(pm->fn[FUNC_X], "");
    pm->dwn[SOURCE] = 0;
    pm->dwr[SOURCE] = 0.0f;
    pm->dwn[TARGET] = 0;
    pm->dwr[TARGET] = 0.0f;
    while ((opt = getopt(
                argc, argv,
                "X:Y:D:z:u:r:w:l:b:k:g:d:e:c:n:N:G:J:K:o:x:y:f:s:hpqvaAtWS1"))
        != -1) {
        switch (opt) {
        case 'D':
            scan_dwpm(pm->dwn, pm->dwr, optarg);
            break;
        case 'G':
            scan_kernel(pm, optarg);
            break;
        case 'z':
            pm->eps = atof(optarg);
            break;
        case 'b':
            pm->bet = atof(optarg);
            break;
        case 'w':
            pm->omg = atof(optarg);
            break;
        case 'l':
            pm->lmd = atof(optarg);
            break;
        case 'k':
            pm->kpa = atof(optarg);
            break;
        case 'g':
            pm->gma = atof(optarg);
            break;
        case 'd':
            pm->dlt = atof(optarg);
            break;
        case 'e':
            pm->lim = atof(optarg);
            break;
        case 'f':
            pm->btn = atof(optarg);
            break;
        case 'c':
            pm->cnv = atof(optarg);
            break;
        case 'n':
            pm->nlp = atoi(optarg);
            break;
        case 'N':
            pm->llp = atoi(optarg);
            break;
        case 'K':
            pm->K = atoi(optarg);
            break;
        case 'J':
            pm->J = atoi(optarg);
            break;
        case 'r':
            pm->rns = atoi(optarg);
            break;
        case 'u':
            pm->nrm = *optarg;
            break;
        case 'h':
            pm->opt |= PW_OPT_HISTO;
            break;
        case 'a':
            pm->opt |= PW_OPT_DBIAS;
            break;
        case 'p':
            pm->opt |= PW_OPT_LOCAL;
            break;
        case 'q':
            pm->opt |= PW_OPT_QUIET;
            break;
        case 'A':
            pm->opt |= PW_OPT_ACCEL;
            break;
        case 'W':
            pm->opt |= PW_OPT_NWARN;
            break;
        case 'S':
            pm->opt |= PW_OPT_NOSIM;
            break;
        case '1':
            pm->opt |= PW_OPT_1NN;
            break;
        case 'o':
            strcpy(pm->fn[OUTPUT], optarg);
            break;
        case 'x':
            strcpy(pm->fn[TARGET], optarg);
            break;
        case 'y':
            strcpy(pm->fn[SOURCE], optarg);
            break;
        case 'X':
            strcpy(pm->fn[FUNC_X], optarg);
            break;
        case 'Y':
            strcpy(pm->fn[FUNC_Y], optarg);
            break;
        case 'v':
            printUsage();
            exit(EXIT_SUCCESS);
            break;
        case 's':
            if (strchr(optarg, 'A'))
                pm->opt |= PW_OPT_SAVE;
            if (strchr(optarg, 'x'))
                pm->opt |= PW_OPT_SAVEX;
            if (strchr(optarg, 'y'))
                pm->opt |= PW_OPT_SAVEY;
            if (strchr(optarg, 'u'))
                pm->opt |= PW_OPT_SAVEU;
            if (strchr(optarg, 'v'))
                pm->opt |= PW_OPT_SAVEV;
            if (strchr(optarg, 'a'))
                pm->opt |= PW_OPT_SAVEA;
            if (strchr(optarg, 'c'))
                pm->opt |= PW_OPT_SAVEC;
            if (strchr(optarg, 'e'))
                pm->opt |= PW_OPT_SAVEE;
            if (strchr(optarg, 'S'))
                pm->opt |= PW_OPT_SAVES;
            if (strchr(optarg, 'P'))
                pm->opt |= PW_OPT_SAVEP;
            if (strchr(optarg, 'T'))
                pm->opt |= PW_OPT_SAVET;
            if (strchr(optarg, 'X'))
                pm->opt |= PW_OPT_PATHX;
            if (strchr(optarg, 'Y'))
                pm->opt |= PW_OPT_PATHY;
            if (strchr(optarg, 't'))
                pm->opt |= PW_OPT_PFLOG;
            if (strchr(optarg, '0'))
                pm->opt |= PW_OPT_VTIME;
            break;
        }
    }
    /* acceleration with default parameters */
    if (pm->opt & PW_OPT_ACCEL) {
        pm->J = 300;
        pm->K = 70;
        pm->opt |= PW_OPT_LOCAL;
    }
    /* case: save all */
    if (pm->opt & PW_OPT_SAVE)
        pm->opt |= PW_OPT_SAVEX | PW_OPT_SAVEU | PW_OPT_SAVEC | PW_OPT_SAVEP | PW_OPT_SAVEA | PW_OPT_PFLOG | PW_OPT_SAVEY | PW_OPT_SAVEV | PW_OPT_SAVEE | PW_OPT_SAVET | PW_OPT_SAVES | PW_OPT_PATHY;
    /* always save y & info */
    pm->opt |= PW_OPT_SAVEY | PW_OPT_INFO;
    /* for numerical stability */
    pm->omg = pm->omg == 0 ? 1e-250 : pm->omg;
    /* llp is always less than or equal to nlp */
    if (pm->llp > pm->nlp)
        pm->llp = pm->nlp;

    return;
}

void memsize(int* dsz, int* isz, pwsz sz, pwpm pm)
{
    int M = sz.M, N = sz.N, J = sz.J, K = sz.K, D = sz.D;
    int T = pm.opt & PW_OPT_LOCAL;
    int L = M > N ? M : N, mtd = MAXTREEDEPTH;
    *isz = D;
    *dsz = 4 * M + 2 * N + D * (5 * M + N + 13 * D + 3); /* common          */
    *isz += K ? M : 0;
    *dsz += K ? K * (2 * M + 3 * K + D + 12) : (3 * M * M); /* low-rank        */
    *isz += J ? (M + N) : 0;
    *dsz += J ? (D * (M + N + J) + J + J * J) : 0; /* nystrom         */
    *dsz += J * (1 + D + 1); /* nystrom (Df=1)  */
    *isz += T ? L * 6 : 0;
    *dsz += T ? L * 2 : 0; /* kdtree (build)  */
    *isz += T ? L * (2 + mtd) : 0; /* kdtree (search) */
    *isz += T ? 2 * (3 * L + 1) : 0; /* kdtree (tree)   */
    *dsz += M; /* function reg    */
}

void print_bbox(const double* X, int D, int N)
{
    int d, n;
    double max, min;
    char ch[3] = { 'x', 'y', 'z' };
    for (d = 0; d < D; d++) {
        max = X[d];
        for (n = 1; n < N; n++)
            max = fmax(max, X[d + D * n]);
        min = X[d];
        for (n = 1; n < N; n++)
            min = fmin(min, X[d + D * n]);
        fprintf(stderr, "%c=[%.2f,%.2f]%s", ch[d], min, max,
            d == D - 1 ? "\n" : ", ");
    }
}

void print_norm(const double* X, const double* Y, int D, int N, int M, int sw,
    char type)
{
    int t = 0;
    char name[4][64] = { "for each", "using X", "using Y", "skipped" };
    switch (type) {
    case 'e':
        t = 0;
        break;
    case 'x':
        t = 1;
        break;
    case 'y':
        t = 2;
        break;
    case 'n':
        t = 3;
        break;
    }
    if (sw) {
        fprintf(stderr, "  Normalization: [%s]\n", name[t]);
        fprintf(stderr, "    Bounding boxes that cover point sets:\n");
    }
    fprintf(stderr, "    %s:\n", sw ? "Before" : "After");
    fprintf(stderr, "      Target: ");
    print_bbox(X, D, N);
    fprintf(stderr, "      Source: ");
    print_bbox(Y, D, M);
    if (!sw)
        fprintf(stderr, "\n");
}

double tvcalc(const LARGE_INTEGER* end, const LARGE_INTEGER* beg)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    return (double)(end->QuadPart - beg->QuadPart) / freq.QuadPart;
}

void fprint_comptime(FILE* fp, const LARGE_INTEGER* tv, double* tt, int nx,
    int ny, int geok)
{
    if (fp == stderr)
        fprintf(fp, "  Computing Time:\n");
#ifdef MINGW32
    if (geok)
        fprintf(fp, "    FPSA algorithm:  %.3lf s\n", tvcalc(tv + 2, tv + 1));
    if (nx || ny)
        fprintf(fp, "    Downsampling:    %.3lf s\n", tvcalc(tv + 3, tv + 2));
    fprintf(fp, "    VB Optimization: %.3lf s\n", tvcalc(tv + 4, tv + 3));
    if (ny)
        fprintf(fp, "    Interpolation:   %.3lf s\n", tvcalc(tv + 5, tv + 4));
#else
    if (geok)
        fprintf(fp, "    FPSA algorithm:  %.3lf s (real) / %.3lf s (cpu)\n",
            tvcalc(tv + 2, tv + 1), (tt[2] - tt[1]) / CLOCKS_PER_SEC);
    if (nx || ny)
        fprintf(fp, "    Downsampling:    %.3lf s (real) / %.3lf s (cpu)\n",
            tvcalc(tv + 3, tv + 2), (tt[3] - tt[2]) / CLOCKS_PER_SEC);
    fprintf(fp, "    VB Optimization: %.3f s (real) / %.3lf s (cpu)\n",
        tvcalc(tv + 4, tv + 3), (tt[4] - tt[3]) / CLOCKS_PER_SEC);
    if (ny)
        fprintf(fp, "    Interpolation:   %.3lf s (real) / %.3lf s (cpu)\n",
            tvcalc(tv + 5, tv + 4), (tt[5] - tt[4]) / CLOCKS_PER_SEC);
#endif
    fprintf(fp, "    File reading:    %.3lf s\n", tvcalc(tv + 1, tv + 0));
    fprintf(fp, "    File writing:    %.3lf s\n", tvcalc(tv + 6, tv + 5));
    if (fp == stderr)
        fprintf(fp, "\n");
}

void fprint_comptime2(FILE* fp, const struct timeval* tv, double* tt,
    int geok)
{
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 2, tv + 1),
        (tt[2] - tt[1]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 3, tv + 2),
        (tt[3] - tt[2]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 4, tv + 3),
        (tt[4] - tt[3]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 5, tv + 4),
        (tt[5] - tt[4]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 1, tv + 0), tvcalc(tv + 6, tv + 5));
}

int main(int argc, char** argv)
{
    int d, k, l, m, n; // ループカウンター
    int D, M, N, lp; // D:次元数，M:点群Xの点数，N:点群Yの点数
    char mode; // ファイル読込モード
    double s, r, Np, sgmX, sgmY, *muX,
        *muY; // s:スケール，r:変形粗さ，Np:推定される点の数，sgmX,sgmY:標準偏差，スケールの調整に使用，muX,muY:平均ベクトル
    double *u, *v, *w; // u,v:変形ベクトル，w:重み
    double **R, *t, *a,
        *sgm; // R:回転ベクトル，t:平行移動ベクトル，a:各点の対応確率，sgm:各点のスケール変化
    pwpm pm; // アルゴのパラメータを格納する構造体
    pwsz sz; // サイズや次元数を格納する構造体
    double *x, *y, *X, *Y, *wd, **bX,
        **bY; // x,y:変換後の点群，X,Y:元の点群，wd:作業用データを格納，bX,bY:ファイルから読み込んだ生の点群
    int* wi; // 作業用の整数データを格納
    int sd = sizeof(double),
        si = sizeof(
            int); // sd,si:double,int型の変数がメモリ上で占めるサイズをバイト単位で保持，基本はそれぞれ8Byte,4Byte
    FILE* fp;
    char fn[256];
    int dsz,
        isz; // dsz,isz:double,int型のデータメモリサイズ・アルゴに必要な浮動小数点数・整数データの総量
    int xsz,
        ysz; // x,y配列に必要なメモリサイズ，点群の次元と点の数、およびアルゴリズムのオプションによってサイズが変化
    char *ytraj = ".optpath.bin", *xtraj = ".optpathX.bin";
    double tt[7]; // tt:各処理の時間を記録
    LARGE_INTEGER tv[7]; // 時間計測用の変数
    int nx, ny, N0,
        M0 = 0; // nx,ny:ターゲット・ソースのダウンサンプリング時の点数，N0,M0:元の点の数
    double rx, ry, *T, *X0,
        *Y0 = NULL; // rx,ry:ダウンサンプリングの比率，T:変換後の点群，X0,Y0:ダウンサンプリング前の点群
    double sgmT,
        *muT; // sgmT:変換後の点群の標準偏差，muT:変換後の点群の平均ベクトル
    double* pf; // pf:アルゴの性能を記録
    double *LQ = NULL,
           *LQ0 = NULL; // LQ,LQ0:ジオデジックカーネル分解の結果を格納する配列
    int *Ux,
        *Uy; // Ux,Uy:ダウンサンプリング時に元点群のどの点がサンプリングされたかを示すindex．ex:ダウンサンプリング後のターゲット点群のi番の点が元点群のUx[i]番の点に対応
    int K; // Geodesic Kernelの形状表現に必要な基底ベクトルの数
    int geok = 0; // geok:ジオデジックカーネルを使用するかどうかのフラグ
    double*
        x0; // x0:補間やその他の後処理で使用される，変換後の点群データを格納するための配列

    QueryPerformanceCounter(tv + 0);
    tt[0] = clock();

    /* ファイルの読み込み */
    pw_getopt(&pm, argc, argv);
    bX = read2d(&N, &D, &mode, pm.fn[TARGET], "NA");
    X = calloc((size_t)D * N, sd);
    sz.D = D;
    bY = read2d(&M, &D, &mode, pm.fn[SOURCE], "NA");
    Y = calloc((size_t)D * M, sd);

    /* 乱数の初期化 */
    init_genrand64(pm.rns ? pm.rns : clock());

    /* 次元数の確認 */
    if (D != sz.D) {
        printf(
            "ERROR: Dimensions of X and Y are incosistent. dim(X)=%d, dim(Y)=%d\n",
            sz.D, D);
        exit(EXIT_FAILURE);
    }
    if (N <= D || M <= D) {
        printf("ERROR: #points must be greater than dimension\n");
        exit(EXIT_FAILURE);
    }

    /* メモリレイアウトの変更 */
    // 1次元配列[x1, x2, x3, ..., xn, y1, y2, y3, ..., yn, z1, z2, z3, ...,
    // zn]に変更
    for (d = 0; d < D; d++)
        for (n = 0; n < N; n++) {
            X[d + D * n] = bX[n][d];
        }
    free2d(bX, N);
    for (d = 0; d < D; d++)
        for (m = 0; m < M; m++) {
            Y[d + D * m] = bY[m][d];
        }
    free2d(bY, M);

    /* alias: size */
    sz.M = M;
    sz.J = pm.J;
    sz.N = N;
    sz.K = pm.K;

    /* check parameters */
    check_prms(pm, sz);

    /* print: paramters */
    if (!(pm.opt & PW_OPT_QUIET))
        printInfo(sz, pm);

    /* 位置，スケールの正規化 */
    muX = calloc(D, sd);
    muY = calloc(D, sd);
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3))
        print_norm(X, Y, D, N, M, 1, pm.nrm);
    normalize_batch(X, muX, &sgmX, Y, muY, &sgmY, N, M, D, pm.nrm);
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3))
        print_norm(X, Y, D, N, M, 0, pm.nrm);

    /*Geodesic Kernelの計算 */
    QueryPerformanceCounter(tv + 1);
    tt[1] = clock();
    // nnk:近隣の点の数を指定するパラメータ，pm.fn[FACE_Y]:メッシュの面情報を含むファイル名，pm.tau:Geodesic
    // Kernelの計算に使用される閾値
    geok = (pm.nnk || strlen(pm.fn[FACE_Y])) && pm.tau > 1e-5;
    // Fast Point Set Alignment
    if (geok && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr, "  Executing the FPSA algorithm ... ");
    if (geok) {
        sgraph* sg;
        if (pm.nnk)
            sg = sgraph_from_points(Y, D, M, pm.nnk,
                pm.nnr); // 点群からスパースグラフを構築
        else
            sg = sgraph_from_mesh(Y, D, M,
                pm.fn[FACE_Y]); // メッシュからスパースグラフを構築
        // スパースグラフ上でGeodesic Kernel分解を行い，その結果を格納する．
        // スパースグラフのエッジ情報sg->Eと重みsg->Wを使用し、パラメータとしてpm.K（基底の数）、pm.bet、pm.tau、pm.epsを受け取る．
        // pm.bet:変形の滑らかさや剛性を制御するパラメータ．大きいほど、より滑らかな変形が促され、小さいほど局所的な変形が許容される．
        // pm.tau:距離の影響を制御する閾値やスケールパラメータ．小さいほど、近い点同士の関係が強調され、大きいほど遠い点同士の関係も考慮に入れられる．
        // pm.eps:反復計算における収束判定
        LQ = geokdecomp(&K, Y, D, M, (const int**)sg->E, (const double**)sg->W,
            pm.K, pm.bet, pm.tau, pm.eps);
        sz.K = pm.K = K; /* update K */
        sgraph_free(sg);
        if (geok && !(pm.opt & PW_OPT_QUIET))
            fprintf(stderr, "done. (K->%d)\n\n", K);
    }

    QueryPerformanceCounter(tv + 2);
    tt[2] = clock();

    nx = pm.dwn[TARGET];
    rx = pm.dwr[TARGET]; // ダウンサンプリング目標点数と比率
    ny = pm.dwn[SOURCE];
    ry = pm.dwr[SOURCE]; // ダウンサンプリング目標点数と比率
    if ((nx || ny) && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr, "  Downsampling ...");
    if (nx) {
        X0 = X;
        N0 = N;
        N = sz.N = nx;
        X = calloc((size_t)D * N, sd);
        Ux = calloc((size_t)D * N, sd);
        downsample(X, Ux, N, X0, D, N0, rx);
    }
    if (ny) {
        Y0 = Y;
        M0 = M;
        M = sz.M = ny;
        Y = calloc((size_t)D * M, sd);
        Uy = calloc((size_t)D * M, sd);
        downsample(Y, Uy, M, Y0, D, M0, ry);
    }
    if ((nx || ny) && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr, " done. \n\n");
    //ダウンサンプリングされた各点に対応するジオデジックカーネルの値を、新しいLQ配列に適用する．
    if (ny && geok) {
        LQ0 = LQ;
        LQ = calloc((size_t)K + (size_t)K * M, sd);
        for (k = 0; k < K; k++)
            LQ[k] = LQ0[k];
        for (k = 0; k < K; k++)
            for (m = 0; m < M; m++)
                LQ[m + M * k + K] = LQ0[Uy[m] + M0 * k + K];
    }

    QueryPerformanceCounter(tv + 3);
    tt[3] = clock();

    /* memory size */
    memsize(&dsz, &isz, sz, pm);

    /* memory size: x, y */
    ysz = D * M;
    ysz += D * M * ((pm.opt & PW_OPT_PATHY) ? pm.nlp : 0);
    xsz = D * M;
    xsz += D * M * ((pm.opt & PW_OPT_PATHX) ? pm.nlp : 0);

    /* allocaltion */
    wd = calloc(dsz, sd);
    x = calloc(xsz, sd);
    a = calloc(M, sd);
    u = calloc((size_t)D * M, sd);
    R = calloc((size_t)D * D, sd);
    sgm = calloc(M, sd);
    wi = calloc(isz, si);
    y = calloc(ysz, sd);
    w = calloc(M, sd);
    v = calloc((size_t)D * M, sd);
    t = calloc(D, sd);
    pf = calloc((size_t)3 * pm.nlp, sd);

    /* main computation */
    lp = bcpd(x, y, u, v, w, a, sgm, &s, R, t, &r, &Np, pf, wd, wi, X, Y, LQ, sz,
        pm);

    /* interpolation */

    QueryPerformanceCounter(tv + 4);
    tt[4] = clock();

    if (ny) {
        if (!(pm.opt & PW_OPT_QUIET))
            fprintf(stderr, "%s  Interpolating ... ",
                (pm.opt & PW_OPT_HISTO) ? "\n" : "");
        T = calloc((size_t)D * M0, sd);
        if (pm.opt & PW_OPT_1NN)
            interpolate_1nn(T, Y0, M0, v, Y, &s, R, t, sz, pm);
        else if (LQ0)
            interpolate_geok(T, Y0, M0, x, Y, w, &s, R, t, &r, LQ0, Uy, sz, pm);
        else
            interpolate(T, Y0, M0, x, Y, w, &s, R, t, &r, sz, pm);
        switch (pm.nrm) {
        case 'e':
            sgmT = sgmX;
            muT = muX;
            break;
        case 'x':
            sgmT = sgmX;
            muT = muX;
            break;
        case 'y':
            sgmT = sgmY;
            muT = muY;
            break;
        case 'n':
            sgmT = 1.0f;
            muT = NULL;
            break;
        default:
            sgmT = sgmX;
            muT = muX;
        }
        denormlize(T, muT, sgmT, M0, D);
        if (!(pm.opt & PW_OPT_QUIET))
            fprintf(stderr, "done. \n\n");
        if (pm.opt & PW_OPT_INTPX) {
            x0 = calloc((size_t)D * M0, sd);
            N0 = nx ? N0 : N;
            interpolate_x(x0, T, X0, D, M0, N0, r, pm);
            denormlize(x0, muT, sgmT, M0, D);
        }
    }

    QueryPerformanceCounter(tv + 5);
    tt[5] = clock();

    /* save interpolated variables */
    if (ny) {
        save_variable(pm.fn[OUTPUT], "y.interpolated.txt", T, D, M0, "%lf",
            TRANSPOSE);
        if (!(pm.opt & PW_OPT_INTPX))
            goto skip;
        save_variable(pm.fn[OUTPUT], "x.interpolated.txt", x0, D, M0, "%lf",
            TRANSPOSE);
        free(x0);
    skip:
        free(T);
    }

    /* save correspondence */
    if ((pm.opt & PW_OPT_SAVEP) | (pm.opt & PW_OPT_SAVEC) | (pm.opt & PW_OPT_SAVEE))
        save_corresp(pm.fn[OUTPUT], X, y, a, sgm, s, r, sz, pm);

    /* save trajectory */
    if (pm.opt & PW_OPT_PATHX)
        save_optpath(xtraj, x + (size_t)D * M, X, sz, pm, lp);
    if (pm.opt & PW_OPT_PATHY)
        save_optpath(ytraj, y + (size_t)D * M, X, sz, pm, lp);

    /* revert normalization */
    denormalize_batch(x, muX, sgmX, y, muY, sgmY, M, M, D, pm.nrm);

    /* save variables */
    if (pm.opt & PW_OPT_SAVEY)
        save_variable(pm.fn[OUTPUT], "y.txt", y, D, M, "%lf", TRANSPOSE);
    if (pm.opt & PW_OPT_SAVEX)
        save_variable(pm.fn[OUTPUT], "x.txt", x, D, M, "%lf", TRANSPOSE);
    if (pm.opt & PW_OPT_SAVEU)
        save_variable(pm.fn[OUTPUT], "u.txt", u, D, M, "%lf", TRANSPOSE);
    if (pm.opt & PW_OPT_SAVEV)
        save_variable(pm.fn[OUTPUT], "v.txt", v, D, M, "%lf", TRANSPOSE);
    if (pm.opt & PW_OPT_SAVEA)
        save_variable(pm.fn[OUTPUT], "a.txt", a, M, 1, "%e", ASIS);
    if (pm.opt & PW_OPT_SAVET) {
        save_variable(pm.fn[OUTPUT], "s.txt", &s, 1, 1, "%lf", ASIS);
        save_variable(pm.fn[OUTPUT], "R.txt", R, D, D, "%lf", ASIS);
        save_variable(pm.fn[OUTPUT], "t.txt", t, D, 1, "%lf", ASIS);
    }
    if ((pm.opt & PW_OPT_SAVEU) | (pm.opt & PW_OPT_SAVEV) | (pm.opt & PW_OPT_SAVET)) {
        save_variable(pm.fn[OUTPUT], "normX.txt", X, D, N, "%lf", TRANSPOSE);
        save_variable(pm.fn[OUTPUT], "normY.txt", Y, D, M, "%lf", TRANSPOSE);
    }
    if ((pm.opt & PW_OPT_SAVES) && (pm.opt & PW_OPT_DBIAS)) {
        for (m = 0; m < M; m++)
            sgm[m] = sqrt(sgm[m]);
        save_variable(pm.fn[OUTPUT], "Sigma.txt", sgm, M, 1, "%e", ASIS);
    }
    QueryPerformanceCounter(tv + 6);
    tt[6] = clock();

    /* output: computing time */
    if (!(pm.opt & PW_OPT_QUIET))
        fprint_comptime(stderr, tv, tt, nx, ny, geok);

    /* save total computing time */
    strcpy(fn, pm.fn[OUTPUT]);
    strcat(fn, "comptime.txt");
    fp = fopen(fn, "w");
    if (pm.opt & PW_OPT_VTIME)
        fprint_comptime2(fp, tv, tt, geok);
    else
        fprint_comptime(fp, tv, tt, nx, ny, geok);
    fclose(fp);

    /* save computing time for each loop */
    if (pm.opt & PW_OPT_PFLOG) {
        strcpy(fn, pm.fn[OUTPUT]);
        strcat(fn, "profile_time.txt");
        fp = fopen(fn, "w");
#ifdef MINGW32
        for (l = 0; l < lp; l++) {
            fprintf(fp, "%f\n", pf[l]);
        }
#else
        for (l = 0; l < lp; l++) {
            fprintf(fp, "%f\t%f\n", pf[l], pf[l + pm.nlp]);
        }
#endif
        fclose(fp);
        strcpy(fn, pm.fn[OUTPUT]);
        strcat(fn, "profile_sigma.txt");
        fp = fopen(fn, "w");
        for (l = 0; l < lp; l++) {
            fprintf(fp, "%f\n", pf[l + 2 * pm.nlp]);
        }
    }

    /* output info */
    if (pm.opt & PW_OPT_INFO) {
        strcpy(fn, pm.fn[OUTPUT]);
        strcat(fn, "info.txt");
        fp = fopen(fn, "w");
        fprintf(fp, "loops\t%d\n", lp);
        fprintf(fp, "sigma\t%lf\n", r);
        fprintf(fp, "N_hat\t%lf\n", Np);
        fclose(fp);
    }
    if ((pm.opt & PW_OPT_PATHY) && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr,
            "  ** Search path during optimization was saved to: [%s]\n\n",
            ytraj);

    free(x);
    free(X);
    free(wd);
    free(muX);
    free(a);
    free(v);
    free(sgm);
    free(y);
    free(Y);
    free(wi);
    free(muY);
    free(u);
    free(R);
    free(t);
    free(pf);

    SetDllDirectory(NULL);

    return 0;
}

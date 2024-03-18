#include "pch.h"

void save_variable(const char *prefix, const char *suffix, const double *var, int D, int J, const char *fmt, int trans) {
    int d, j;
    char fn[256];
    double **buf;
    strcpy(fn, prefix);
    strcat(fn, suffix);
    if (trans == TRANSPOSE) {
        buf = calloc2d(J, D);
        for (j = 0; j < J; j++)
            for (d = 0; d < D; d++)
                buf[j][d] = var[d + D * j];
        write2d(fn, (const double **)buf, J, D, fmt, "NA");
        free2d(buf, J);
    } else {
        buf = calloc2d(D, J);
        for (j = 0; j < J; j++)
            for (d = 0; d < D; d++)
                buf[d][j] = var[d + D * j];
        write2d(fn, (const double **)buf, D, J, fmt, "NA");
        free2d(buf, D);
    }

    return;
}

void save_corresp(const char *prefix, const double *X, const double *y, const double *a, const double *sgm, const double s, const double r, pwsz sz,
                  pwpm pm) {
    int i, m, n, D, M, N;
    int *T, *l, *bi;
    double *bd;
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

    T = static_cast<int *>(calloc(3 * M + 1, si));
    bi = static_cast<int *>(calloc(6 * M, si));
    bd = static_cast<double *>(calloc(2 * M, sd));
    p = static_cast<double *>(calloc(M, sd));
    l = static_cast<int *>(calloc(M, si));
    kdtree(T, bi, bd, y, D, M);
    vol = volume(X, D, N);
    c = (pow(2.0 * M_PI * SQ(r), 0.5 * D) * omg) / (vol * (1 - omg));
    for (n = 0; n < N; n++) {
        /* compute P, c, e */
        val = c;
        top = ct = 0;
        do {
            eballsearch_next(&m, S, &top, X + D * n, rad, y, T, D, M);
            if (m >= 0)
                l[ct++] = m;
        } while (top);
        if (!ct) {
            nnsearch(&m, &min, X + D * n, y, T, D, M);
            l[ct++] = m;
        }
        for (i = 0; i < ct; i++) {
            m = l[i];
            p[i] = a[m] * gauss(y + D * m, X + D * n, D, r) * (db ? exp(-0.5 * D * SQ(sgm[m] * s / r)) : 1.0);
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
            fprintf(fpe, "%d\t%d\t%lf\n", n + 1, mmax ? mmax : l[0], mmax ? max : p[0]);
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

int save_optpath(const char *file, const double *sy, const double *X, pwsz sz, pwpm pm, int lp) {
    int N = sz.N, M = sz.M, D = sz.D;
    int si = sizeof(int), sd = sizeof(double);
    FILE *fp = fopen(file, "wb");
    if (!fp) {
        printf("Can't open: %s\n", file);
        exit(EXIT_FAILURE);
    }
    fwrite(&N, si, 1, fp);
    fwrite(&D, si, 1, fp);
    fwrite(&M, si, 1, fp);
    fwrite(&lp, si, 1, fp);
    fwrite(sy, sd, lp * D * M, fp);
    fwrite(X, sd, D * N, fp);
    if (strlen(pm.fn[FACE_Y])) {
        double **b;
        int nl, nc, l, c;
        char mode;
        int *L = NULL;
        b = read2d(&nl, &nc, &mode, pm.fn[FACE_Y], "NA");
        assert(nc == 3 || nc == 2);
        L = static_cast<int *>(calloc(nc * nl, si));
        for (l = 0; l < nl; l++)
            for (c = 0; c < nc; c++) {
                L[c + nc * l] = (int)b[l][c];
            }
        fwrite(&nl, si, 1, fp);
        fwrite(&nc, si, 1, fp);
        fwrite(L, si, nc * nl, fp);
        free(L);
        free(b);
    }
    fclose(fp);

    return 0;
}

void scan_kernel(pwpm *pm, char *arg) {
    char *p;
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
        printf("ERROR: the 2nd argument of -G (tau) must be in range [0,1]. "
               "Abort.\n");
        exit(EXIT_FAILURE);
    }
}

void scan_dwpm(int *dwn, double *dwr, const char *arg) {
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

void check_prms(const pwpm pm, const pwsz sz) {
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
        printf("\n  ERROR: -u: Argument must be one of 'e', 'x', 'y' and 'n'. "
               "Abort.\n\n");
        exit(EXIT_FAILURE);
    }
}

void pw_getopt(pwpm *pm, int argc, char **argv) {
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
    while ((opt = getopt(argc, argv, "X:Y:D:z:u:r:w:l:b:k:g:d:e:c:n:N:G:J:K:o:x:y:f:s:hpqvaAtWS1")) != -1) {
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
        pm->opt |= PW_OPT_SAVEX | PW_OPT_SAVEU | PW_OPT_SAVEC | PW_OPT_SAVEP | PW_OPT_SAVEA | PW_OPT_PFLOG | PW_OPT_SAVEY | PW_OPT_SAVEV |
                   PW_OPT_SAVEE | PW_OPT_SAVET | PW_OPT_SAVES | PW_OPT_PATHY;
    /* always save y & info */
    pm->opt |= PW_OPT_SAVEY | PW_OPT_INFO;
    /* for numerical stability */
    pm->omg = pm->omg == 0 ? 1e-250 : pm->omg;
    /* llp is always less than or equal to nlp */
    if (pm->llp > pm->nlp)
        pm->llp = pm->nlp;

    return;
}

void memsize(int *dsz, int *isz, pwsz sz, pwpm pm) {
    int M = sz.M, N = sz.N, J = sz.J, K = sz.K, D = sz.D;
    int T = pm.opt & PW_OPT_LOCAL;
    int L = M > N ? M : N, mtd = MAXTREEDEPTH;
    *isz = D;
    *dsz = 4 * M + 2 * N + D * (5 * M + N + 13 * D + 3); /* common          */
    *isz += K ? M : 0;
    *dsz += K ? K * (2 * M + 3 * K + D + 12) : (3 * M * M); /* low-rank */
    *isz += J ? (M + N) : 0;
    *dsz += J ? (D * (M + N + J) + J + J * J) : 0; /* nystrom         */
    *dsz += J * (1 + D + 1);                       /* nystrom (Df=1)  */
    *isz += T ? L * 6 : 0;
    *dsz += T ? L * 2 : 0;           /* kdtree (build)  */
    *isz += T ? L * (2 + mtd) : 0;   /* kdtree (search) */
    *isz += T ? 2 * (3 * L + 1) : 0; /* kdtree (tree)   */
    *dsz += M;                       /* function reg    */
}

void print_bbox(const double *X, int D, int N) {
    int d, n;
    double max, min;
    char ch[3] = {'x', 'y', 'z'};
    for (d = 0; d < D; d++) {
        max = X[d];
        for (n = 1; n < N; n++)
            max = fmax(max, X[d + D * n]);
        min = X[d];
        for (n = 1; n < N; n++)
            min = fmin(min, X[d + D * n]);
        fprintf(stderr, "%c=[%.2f,%.2f]%s", ch[d], min, max, d == D - 1 ? "\n" : ", ");
    }
}

void print_norm(const double *X, const double *Y, int D, int N, int M, int sw, char type) {
    int t = 0;
    char name[4][64] = {"for each", "using X", "using Y", "skipped"};
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

double tvcalc(const LARGE_INTEGER *end, const LARGE_INTEGER *beg) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    return (double)(end->QuadPart - beg->QuadPart) / freq.QuadPart;
}

void fprint_comptime(FILE *fp, const LARGE_INTEGER *tv, double *tt, int nx, int ny, int geok) {
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
        fprintf(fp, "    FPSA algorithm:  %.3lf s (real) / %.3lf s (cpu)\n", tvcalc(tv + 2, tv + 1), (tt[2] - tt[1]) / CLOCKS_PER_SEC);
    if (nx || ny)
        fprintf(fp, "    Downsampling:    %.3lf s (real) / %.3lf s (cpu)\n", tvcalc(tv + 3, tv + 2), (tt[3] - tt[2]) / CLOCKS_PER_SEC);
    fprintf(fp, "    VB Optimization: %.3f s (real) / %.3lf s (cpu)\n", tvcalc(tv + 4, tv + 3), (tt[4] - tt[3]) / CLOCKS_PER_SEC);
    if (ny)
        fprintf(fp, "    Interpolation:   %.3lf s (real) / %.3lf s (cpu)\n", tvcalc(tv + 5, tv + 4), (tt[5] - tt[4]) / CLOCKS_PER_SEC);
#endif
    fprintf(fp, "    File reading:    %.3lf s\n", tvcalc(tv + 1, tv + 0));
    fprintf(fp, "    File writing:    %.3lf s\n", tvcalc(tv + 6, tv + 5));
    if (fp == stderr)
        fprintf(fp, "\n");
}

void fprint_comptime2(FILE *fp, const LARGE_INTEGER *tv, double *tt, int geok) {
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 2, tv + 1), (tt[2] - tt[1]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 3, tv + 2), (tt[3] - tt[2]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 4, tv + 3), (tt[4] - tt[3]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 5, tv + 4), (tt[5] - tt[4]) / CLOCKS_PER_SEC);
    fprintf(fp, "%lf\t%lf\n", tvcalc(tv + 1, tv + 0), tvcalc(tv + 6, tv + 5));
}

void free_all(void *ptr, ...) {
    va_list args;
    va_start(args, ptr);

    while (ptr != NULL) {
        free(ptr);
        ptr = va_arg(args, void *);
    }

    va_end(args);
}

bool loadModel(const char *path, int &numOfPts, int &dim, Eigen::MatrixXd &verts, Eigen::MatrixXi &faces, bool debug) {
    // とりあえずPLY実装
    // TODO: OBJ, Other format
    std::string path_str = std::string(path);
    bool success = false;
    success = igl::readPLY(path_str, verts, faces);
    if (debug) {
        std::cout << "number of mesh.V is " << verts.rows() << '\n';
        std::cout << "number of mesh.F is " << faces.rows() << '\n';
        std::cout << "dimension of mesh.V is " << verts.cols() << '\n';
        std::cout << "dimension of mesh.F is " << faces.cols() << '\n';
    }
    numOfPts = verts.rows();
    dim = verts.cols();
    return success;
}

void changeMemoryLayout(Eigen::MatrixXd &verts, Eigen::MatrixXi &faces, double *&verts_array, int *&faces_array) {
    int N = verts.rows();
    int D = verts.cols();
    verts_array = new double[N * D];
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            verts_array[n * D + d] = verts(n, d);
        }
    }

    int F = faces.rows();
    int V = faces.cols();
    faces_array = new int[F * V];
    for (int f = 0; f < F; f++) {
        for (int v = 0; v < V; v++) {
            faces_array[f * V + v] = faces(f, v);
        }
    }
}

void calculatePrincipalCurvature(const Eigen::MatrixXd &verts, const Eigen::MatrixXi &faces, CurvatureInfo &curvature, const std::string method) {
    // Gaussian curvature, Mean curvature and so on represent local properties of a surface, so use them for different purposes
    // Mean Curvature : (> 0) = 凸-plane, ( = 0) = plane or saddle, (< 0) : 凹-plane
    // Gaussian Curvature : (>0) = dome-like, ( = 0) = plane, (< 0 ) = : saddle
    igl::principal_curvature(verts, faces, curvature.PD1, curvature.PD2, curvature.PV1, curvature.PV2);
    Eigen::MatrixXd H;
    if (method == "gaussian") {
        H = curvature.PD1 * curvature.PD2;
    } else if (method == "mean") {
        H = curvature.PD1 + curvature.PD2;
    } else {
        std::cerr << "Error : No such that method.\n";
    }

    if (H.size() > 0) {
        curvature.Curv = (H.array() - H.minCoeff()) / (H.maxCoeff() - H.minCoeff());
    } else {
        std::cerr << "Error : failed to calculate curvature\n";
    }
}

void visualizeModel(const Eigen::MatrixXd verts, const Eigen::MatrixXi &faces, const Eigen::MatrixXd &feats) {
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(verts, faces);
    if (feats.size() > 0) {
        igl::ColorMapType cmap = igl::COLOR_MAP_TYPE_JET;
        viewer.data().set_data(feats, cmap);
    }
    viewer.launch();
}

sgraph *sgraph_from_mesh_data(const Eigen::MatrixXd &Verts, const Eigen::MatrixXi &Faces) {
    sgraph *sg;
    int M = Verts.rows();
    int D = Verts.cols();

    /* construct graph */
    sg = sgraph_new(M);
    sg->beg = 0;

    for (int l = 0; l < Faces.rows(); l++) {
        int v0, v1, v2;

        /* case: mesh */
        v0 = Faces(l, 0);
        v1 = Faces(l, 1);
        v2 = Faces(l, 2);

        add_uedge(sg, v0, v1, dist(Verts.row(v0).data(), Verts.row(v1).data(), D, 1));
        add_uedge(sg, v1, v2, dist(Verts.row(v1).data(), Verts.row(v2).data(), D, 1));
        add_uedge(sg, v2, v0, dist(Verts.row(v2).data(), Verts.row(v0).data(), D, 1));
    }

    assert(issymmetry((const int **)sg->E, (const double **)sg->W, M));

    return sg;
}

void dump_geokdecomp_output_ply(const char *filename, const double *LQ, const double *Y, int D, int M, int K) {
    std::cout << "----------------------------------- Write Kernel \n";
    FILE *fp = fopen(filename, "w");
    // wだとファイルが無かったら新規作成されるから，わざわざエラーハンドリングしなくて良いのでは？
    //if (fp == NULL) {
    //    fprintf(stderr, "ERROR: Failed to open file %s for writing.\n", filename);
    //    exit(EXIT_FAILURE);
    //}

    // Write PLY header
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", M);
    fprintf(fp, "property double x\n");
    fprintf(fp, "property double y\n");
    fprintf(fp, "property double z\n");
    for (int k = 0; k < K; k++) {
        fprintf(fp, "property double eigenvector%d\n", k);
    }
    fprintf(fp, "element eigenvalue %d\n", K);
    fprintf(fp, "property double value\n");
    fprintf(fp, "end_header\n");

    // Write vertex data
    for (int m = 0; m < M; m++) {
        // Write point coordinates
        for (int d = 0; d < D; d++) {
            fprintf(fp, "%.8f ", Y[m * D + d]);
        }

        // Write eigenvector values for the current point
        for (int k = 0; k < K; k++) {
            fprintf(fp, "%.8f ", LQ[m + M * k + K]);
        }
        fprintf(fp, "\n");
    }

    // Write eigenvalue data
    for (int k = 0; k < K; k++) {
        fprintf(fp, "%.8f\n", LQ[k]);
    }

    fclose(fp);
}
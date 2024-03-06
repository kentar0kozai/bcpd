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
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>
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
// #include "getopt.h/*" /

#define SQ(x) ((x) * (x))
#define M_PI 3.14159265358979323846 // pi

void init_genrand64(unsigned long s);
enum transpose { ASIS = 0,
    TRANSPOSE = 1 };

void scan_kernel(pwpm* pm, const char* arg)
{
    char* p;

    if (std::tolower(*arg) != 'g') {
        pm->G = std::atoi(arg);
        return;
    }

    p = std::strchr(const_cast<char*>(arg), ',');
    if (!p) {
        std::printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");
        std::exit(EXIT_FAILURE);
    }

    if (std::strstr(arg, "txt"))
        std::sscanf(p + 1, "%lf,%s", &(pm->tau), pm->fn[FACE_Y]);
    else
        std::sscanf(p + 1, "%lf,%d,%lf", &(pm->tau), &(pm->nnk), &(pm->nnr));

    if (pm->tau < 0 || pm->tau > 1) {
        std::printf(
            "ERROR: the 2nd argument of -G (tau) must be in range [0,1]. Abort.\n");
        std::exit(EXIT_FAILURE);
    }
}

void scan_dwpm(int* dwn, double* dwr, const std::vector<std::string>& opts)
{
    if (opts.empty() || opts.size() != 3) {
        std::cout << "ERROR: The argument of '-D' must be 'char,int,real'." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    char c = opts[0][0];
    int n = std::stoi(opts[1]);
    double r = std::stod(opts[2]);

    if (n <= 0) {
        std::cout << "ERROR: The 2nd argument of '-D' must be positive." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (r < 0) {
        std::cout << "ERROR: The 3rd argument of '-D' must be positive or 0." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (std::isupper(c)) {
        r *= -1.0f;
    }

    c = std::tolower(c);

    if (c != 'x' && c != 'y' && c != 'b') {
        std::cout << "ERROR: The 1st argument of '-D' must be one of [x,y,b,X,Y,B]." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (r < 0 && -r < 1e-2) {
        r = -1e-2;
    }

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

std::vector<std::string> getopt(int argc, char** argv, const std::string& optstring)
{
    std::vector<std::string> opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg.length() > 1) {
                if (optstring.find(arg[1]) != std::string::npos) {
                    opts.push_back(arg);
                }
            }
        }
    }
    return opts;
}

std::vector<std::string> splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string getOptionValue(const std::vector<std::string>& opts, const std::string& option)
{
    for (const auto& opt : opts) {
        if (opt.substr(0, option.length()) == option) {
            return opt.substr(option.length());
        }
    }
    return "";
}

void pw_getopt(pwpm* pm, int argc, char** argv)
{
    strcpy(pm->fn[TARGET], "X.txt");
    std::cout << pm->fn[TARGET] << std::endl;
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

    std::cout << "argc: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    std::vector<std::string> opts = getopt(argc, argv, "X:Y:D:z:u:r:w:l:b:k:g:d:e:c:n:N:G:J:K:o:x:y:f:s:hpqvaAtWS1");
    if (opts.empty())
        std::cout << "No options provided." << std::endl;
    for (const auto& op : opts) {
        std::cout << "Option : " << op << std::endl;
    }
    // scan_dwpm(pm->dwn, pm->dwr, opts);
    std::string dValue = getOptionValue(opts, "-D");
    if (!dValue.empty()) {
        std::cout << "Value of option 'D': " << dValue << '\n';
        std::vector<std::string> dParams = splitString(dValue.substr(1), ',');
        // TODO: ��������scan_dwpm�̎������Q�l�ɂ��Ď�������
        if (dParams.size() == 3) {
            pm->dwn[SOURCE] = dParams[0] == "B" ? 1 : 0;
            pm->dwr[SOURCE] = std::stof(dParams[1]);
            pm->dwr[TARGET] = std::stof(dParams[2]);
            std::cout << "pm->dwn[SOURCE]: " << pm->dwn[SOURCE] << '\n';
            std::cout << "pm->dwr[SOURCE]: " << pm->dwr[SOURCE] << '\n';
            std::cout << "pm->dwr[TARGET]: " << pm->dwr[TARGET] << '\n';
        } else {
            std::cout << "Invalid format for option 'D'." << '\n';
        }
    } else {
        std::cout << "Option 'D' not found or has no value." << '\n';
    }

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

int main(int argc, char** argv)
{
    int d, k, l, m, n; // ���[�v�J�E���^�[
    int D, M, N, lp; // D:�������CM:�_�QX�̓_���CN:�_�QY�̓_��
    char mode; // �t�@�C���Ǎ����[�h
    double s, r, Np, sgmX, sgmY, *muX, *muY; // s:�X�P�[���Cr:�ό`�e���CNp:���肳���_�̐��CsgmX,sgmY:�W���΍��C�X�P�[���̒����Ɏg�p�CmuX,muY:���σx�N�g��
    double *u, *v, *w; // u,v:�ό`�x�N�g���Cw:�d��
    double **R, *t, *a, *sgm; // R:��]�x�N�g���Ct:���s�ړ��x�N�g���Ca:�e�_�̑Ή��m���Csgm:�e�_�̃X�P�[���ω�
    pwpm pm; // �A���S�̃p�����[�^���i�[����\����
    pwsz sz; // �T�C�Y�⎟�������i�[����\����
    double *x, *y, *X, *Y, *wd, *bX, **bY; // x,y:�ϊ���̓_�Q�CX,Y:���̓_�Q�Cwd:��Ɨp�f�[�^���i�[�CbX,bY:�t�@�C������ǂݍ��񂾐��̓_�Q
    int* wi; // ��Ɨp�̐����f�[�^���i�[
    int sd = sizeof(double), si = sizeof(int); // sd,si:double,int�^�̕ϐ�����������Ő�߂�T�C�Y���o�C�g�P�ʂŕێ��C��{�͂��ꂼ��8Byte,4Byte
    std::FILE* fp;
    char fn[256];
    int dsz, isz; // dsz,isz:double,int�^�̃f�[�^�������T�C�Y�E�A���S�ɕK�v�ȕ��������_���E�����f�[�^�̑���
    int xsz, ysz; // x,y�z��ɕK�v�ȃ������T�C�Y�C�_�Q�̎����Ɠ_�̐��A����уA���S���Y���̃I�v�V�����ɂ���ăT�C�Y���ω�
    char ytraj[] = ".optpath.bin", xtraj[] = ".optpathX.bin";
    double tt[7]; // tt:�e�����̎��Ԃ��L�^
    LARGE_INTEGER tv[7]; // ���Ԍv���p�̕ϐ�
    int nx, ny, N0, M0 = 0; // nx,ny:�^�[�Q�b�g�E�\�[�X�̃_�E���T���v�����O���̓_���CN0,M0:���̓_�̐�
    double rx, ry, *T, *X0, *Y0 = NULL; // rx,ry:�_�E���T���v�����O�̔䗦�CT:�ϊ���̓_�Q�CX0,Y0:�_�E���T���v�����O�O�̓_�Q
    double sgmT, *muT; // sgmT:�ϊ���̓_�Q�̕W���΍��CmuT:�ϊ���̓_�Q�̕��σx�N�g��
    double* pf; // pf:�A���S�̐��\���L�^
    double *LQ = NULL, *LQ0 = NULL; // LQ,LQ0:�W�I�f�W�b�N�J�[�l�������̌��ʂ��i�[����z��
    int *Ux, *Uy; // Ux,Uy:�_�E���T���v�����O���Ɍ��_�Q�̂ǂ̓_���T���v�����O���ꂽ��������index�Dex:�_�E���T���v�����O��̃^�[�Q�b�g�_�Q��i�Ԃ̓_�����_�Q��Ux[i]�Ԃ̓_�ɑΉ�
    int K; // Geodesic Kernel�̌`��\���ɕK�v�Ȋ��x�N�g���̐�
    int geok = 0; // geok:�W�I�f�W�b�N�J�[�l�����g�p���邩�ǂ����̃t���O
    double* x0; // x0:��Ԃ₻�̑��̌㏈���Ŏg�p�����C�ϊ���̓_�Q�f�[�^���i�[���邽�߂̔z��

    QueryPerformanceCounter(tv + 0);
    tt[0] = clock();

    pw_getopt(&pm, argc, argv);
    std::cout << "Hello" << std::endl;
}

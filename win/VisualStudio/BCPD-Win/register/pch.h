#pragma once

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NOMINMAX
#include <windows.h>

extern "C" {
#include "../../../../base/kdtree.h"
#include "../../../../base/kernel.h"
#include "../../../../base/misc.h"
#include "../../../../base/sampling.h"
#include "../../../../base/sgraph.h"
#include "../../../../base/util.h"
#include "../../../../register/bcpd-util.h"
#include "../../../../register/info.h"
#include "../../../../register/norm.h"
#include "getopt.h"
void init_genrand64(unsigned long s);
}

#include "igl/principal_curvature.h"

#define SQ(x) ((x) * (x))
#define M_PI 3.14159265358979323846 // pi

enum transpose { ASIS = 0, TRANSPOSE = 1 };

void save_variable(const char *prefix, const char *suffix, const double *var, int D, int J, const char *fmt, int trans);

void save_corresp(const char *prefix, const double *X, const double *y, const double *a, const double *sgm, const double s, const double r, pwsz sz,
                  pwpm pm);

int save_optpath(const char *file, const double *sy, const double *X, pwsz sz, pwpm pm, int lp);

void scan_kernel(pwpm *pm, char *arg);

void scan_dwpm(int *dwn, double *dwr, const char *arg);

void check_prms(const pwpm pm, const pwsz sz);

void pw_getopt(pwpm *pm, int argc, char **argv);

void memsize(int *dsz, int *isz, pwsz sz, pwpm pm);

void print_bbox(const double *X, int D, int N);

void print_norm(const double *X, const double *Y, int D, int N, int M, int sw, char type);

double tvcalc(const LARGE_INTEGER *end, const LARGE_INTEGER *beg);

void fprint_comptime(FILE *fp, const LARGE_INTEGER *tv, double *tt, int nx, int ny, int geok);

void fprint_comptime2(FILE *fp, const LARGE_INTEGER *tv, double *tt, int geok);
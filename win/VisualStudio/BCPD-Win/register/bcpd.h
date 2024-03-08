#pragma once
#include "pch.h"

int bcpd(double *x,   /*  O  | DM x 1 (+nlp) | aligned target shape */
         double *y,   /*  O  | DM x 1 (+nlp) | deformed source shape */
         double *u,   /*  O  | DM x 1        | normalized def. shape */
         double *v,   /*  O  | DM x 1        | displacement vectors */
         double *w,   /*  O  |  M x 1        | #matches for each m */
         double *a,   /*  O  |  M x 1        | mixing coefficients */
         double *sgm, /*  O  |  M x 1        | posterior covariance */
         double *s, /*  O  |    1          | scale factor */ double *R,
         /*  O  |  D x D        | rotation matrix          */ double *t, /*  O
         |    D          | translation vector       */
         double *r,                                                      /*  O
         |    1          | residual s.d.            */
         double *Np,                                                     /*  O
         |    1          | #matched points (est'd)  */
         double *pf,                                                     /*  O
         | nlp x 3       | comp. time (r/c) & sigma */
         double *wd,                                                     /*  W
         |    *          | working memory (double)  */
         int *wi,                                                        /*  W
         |    *          | working memory (int)     */
         const double *X,                                                /*  I
         | DN x 1        | target point set         */
         const double *Y,                                                /*  I
         | DM x 1        | source point set         */
         const double *LQ,                                               /*  I
         | K + M x K     | only for geodesic kernel */
         const pwsz sz,                                                  /*  I
         |               | D, M, N, K, J            */
         const pwpm pm                                                   /*  I
         |               | tuning parameters        */
);
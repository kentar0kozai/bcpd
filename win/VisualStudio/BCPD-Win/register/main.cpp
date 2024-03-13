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

#include "bcpd.h"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/principal_curvature.h>
#include <igl/readPLY.h>
#include <igl/rgb_to_hsv.h>
#include <stdarg.h>
#include <stdlib.h>
extern "C" {
#include "../../../../base/geokdecomp.h"
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

bool loadModel(const char *path, int *numOfPts, int *dim, Eigen::MatrixXd &verts, Eigen::MatrixXi &faces, bool debug) {
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
    return success;
}

struct CurvatureInfo {
    Eigen::MatrixXd PD1; // Principal curvature direction 1
    Eigen::MatrixXd PD2; // Principal curvature direction 2
    Eigen::VectorXd PV1; // Principal curvature value 1
    Eigen::VectorXd PV2; // Principal curvature value 2
    Eigen::VectorXd Curv;
};

void calculatePrincipalCurvature(const Eigen::MatrixXd &verts, const Eigen::MatrixXi &faces, CurvatureInfo &curvature, const std::string method) {
    // Gaussian curvature, Mean curvature and so on represent local properties of a surface, so use them for different purposes
    // Mean Curvature : (> 0) = 凸-plane, ( = 0) = plane or saddle, (< 0) : 凹-plane
    // Gaussian Curvature : (>0) = dome-like, ( = 0) = plane, (< 0 ) = : saddle
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

void convertToFormat(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                     int &D,     // 次元数
                     int &M,     // 面の数
                     double *&Y, // 点群データ（1次元配列）
                     int **&mesh // メッシュ情報
) {
    int N = V.rows(); // 頂点の数
    D = V.cols();     // 次元数を格納
    M = F.rows();     // 面の数を格納

    // 点群データの整形
    Y = new double[D * N];
    for (int d = 0; d < D; ++d) {
        for (int n = 0; n < N; ++n) {
            Y[d + D * n] = V(n, d);
        }
    }

    // メッシュ情報の整形
    mesh = new int *[M];
    for (int m = 0; m < M; ++m) {
        mesh[m] = new int[3];         // 三角形メッシュを想定
        for (int d = 0; d < 3; ++d) { // 三角形の各頂点
            mesh[m][d] = F(m, d);
        }
    }

    // この時点で、D, M, Y, meshに関数の結果が格納されている
}

void freeMesh(int **&mesh, int M) {
    for (int i = 0; i < M; ++i) {
        delete[] mesh[i];
    }
    delete[] mesh;
    mesh = nullptr;
}

int main(int argc, char **argv) {
    int d, k, l, m, n; // ループカウンター
    int D, M, N, lp;   // D:次元数，M:点群Xの点数，N:点群Yの点数
    char mode;         // ファイル読込モード
    double s, r, Np, sgmX, sgmY, *muX, *muY; // s:スケール，r:変形粗さ，Np:推定点の数，sgmX,sgmY:標準偏差，スケールの調整に使用，muX,muY:平均ベクトル
    double *u, *v, *w;       // u,v:変形ベクトル，w:重み
    double *R, *t, *a, *sgm; // R:回転ベクトル，t:平行移動ベクトル，a:各点の対応確率，sgm:各点のスケール変化
    pwpm pm;                 // アルゴのパラメータを格納する構造体
    pwsz sz;                 // サイズや次元数を格納する構造体
    double *x, *y, *X, *Y, *wd, **bX, **bY; // x,y:変換後の点群，X,Y:元の点群，wd:作業用データを格納，bX,bY:ファイルから読み込んだ生の点群
    int *wi;                                // 作業用の整数データを格納
    int sd = sizeof(double), si = sizeof(int); // sd,si:double,int型の変数がメモリ上で占めるサイズをバイト単位で保持，基本はそれぞれ8Byte,4Byte
    FILE *fp;
    char fn[256];
    int dsz, isz; // dsz,isz:double,int型のデータメモリサイズ・アルゴに必要な浮動小数点数・整数データの総量
    int xsz, ysz; // x,y配列に必要なメモリサイズ，点群の次元と点の数、およびアルゴリズムのオプションによってサイズが変化
    const char *ytraj = ".optpath.bin", *xtraj = ".optpathX.bin";
    double tt[7];                       // tt:各処理の時間を記録
    LARGE_INTEGER tv[7];                // 時間計測用の変数
    int nx, ny, N0, M0 = 0;             // nx,ny:ターゲット・ソースのダウンサンプリング時の点数，N0,M0:元の点の数
    double rx, ry, *T, *X0, *Y0 = NULL; // rx,ry:ダウンサンプリングの比率，T:変換後の点群，X0,Y0:ダウンサンプリング前の点群
    double sgmT, *muT;                  // sgmT:変換後の点群の標準偏差，muT:変換後の点群の平均ベクトル
    double *pf;                         // pf:アルゴの性能を記録
    double *LQ = NULL, *LQ0 = NULL; // LQ,LQ0:ジオデジックカーネル分解の結果を格納する配列
    int *Ux, *Uy; // Ux,Uy:ダウンサンプリング時に元点群indexを保存．ex:ダウンサンプリング後のターゲット点群のi番の点が元点群のUx[i]番の点に対応
    int K;        // Geodesic Kernelの形状表現に必要な基底ベクトルの数
    int geok = 0; // geok:ジオデジックカーネルを使用するかどうかのフラグ
    double *x0;   // x0:補間やその他の後処理で使用される，変換後の点群データを格納するための配列

    /* ファイルの読み込み */
    pw_getopt(&pm, argc, argv);

    // Eigen::MatrixXd V; // Verticies
    // Eigen::MatrixXi F; // Triangles
    // bool success = false;
    // success = loadModel(pm.fn[SOURCE], &N, &D, V, F, true);
    // CurvatureInfo curv_info;
    // std::string curv_method = "gaussian";
    // calculatePrincipalCurvature(V, F, curv_info, curv_method);
    //// visualizeModel(V, F, curv_info.Curv);
    // visualizeModel(V, F, curv_info.PD1 * curv_info.PD2);

    QueryPerformanceCounter(tv + 0);
    tt[0] = clock();

    // bX = read2d(&N, &D, &mode, pm.fn[TARGET], "NA");
    // X = static_cast<double *>(calloc((size_t)D * N, sd));
    // sz.D = D;
    // bY = read2d(&M, &D, &mode, pm.fn[SOURCE], "NA");
    // Y = static_cast<double *>(calloc((size_t)D * M, sd));

    /* メッシュの読み込み */
    Eigen::MatrixXd X_verts, Y_verts;
    Eigen::MatrixXi X_faces, Y_faces;
    CurvatureInfo X_Curv, Y_Curv;
    std::string curv_method = "gaussian";
    bool success = false;
    success = loadModel(pm.fn[SOURCE], &M, &D, Y_verts, Y_faces, true);
    if (!success) {
        throw std::runtime_error("Failed to load source model.");
    }
    success = loadModel(pm.fn[TARGET], &N, &D, X_verts, X_faces, true);
    if (!success) {
        throw std::runtime_error("Failed to load target model.");
    }
    calculatePrincipalCurvature(Y_verts, Y_faces, Y_Curv, curv_method);
    calculatePrincipalCurvature(X_verts, X_faces, X_Curv, curv_method);
    visualizeModel(Y_verts, Y_faces, Y_Curv.Curv);

    /* 乱数の初期化 */
    init_genrand64(pm.rns ? pm.rns : clock());

    /* 次元数の確認 */
    if (D != sz.D) {
        printf("ERROR: Dimensions of X and Y are incosistent. dim(X)=%d, dim(Y)=%d\n", sz.D, D);
        exit(EXIT_FAILURE);
    }
    if (N <= D || M <= D) {
        printf("ERROR: #points must be greater than dimension\n");
        exit(EXIT_FAILURE);
    }

    /* メモリレイアウトの変更 */
    // 1次元配列[x1, x2, x3, ..., xn, y1, y2, y3, ..., yn, z1, z2, z3, ..., zn]に変更
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
    muX = static_cast<double *>(calloc(D, sd));
    muY = static_cast<double *>(calloc(D, sd));
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
        sgraph *sg;
        if (pm.nnk)
            sg = sgraph_from_points(Y, D, M, pm.nnk, pm.nnr); // 点群からスパースグラフを構築
        else
            sg = sgraph_from_mesh(Y, D, M, pm.fn[FACE_Y]); // メッシュからスパースグラフを構築
        // スパースグラフ上でGeodesic Kernel分解を行い，その結果を格納する．
        // スパースグラフのエッジ情報sg->Eと重みsg->Wを使用し、パラメータとしてpm.K（基底の数）、pm.bet、pm.tau、pm.epsを受け取る．
        // pm.bet:変形の滑らかさや剛性を制御するパラメータ．大きいほど、より滑らかな変形が促され、小さいほど局所的な変形が許容される．
        // pm.tau:距離の影響を制御する閾値やスケールパラメータ．小さいほど、近い点同士の関係が強調され、大きいほど遠い点同士の関係も考慮に入れられる．
        // pm.eps:反復計算における収束判定
        LQ = geokdecomp(&K, Y, D, M, (const int **)sg->E, (const double **)sg->W, pm.K, pm.bet, pm.tau, pm.eps);
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
        X = static_cast<double *>(calloc((size_t)D * N, sd));
        Ux = static_cast<int *>(calloc((size_t)D * N, si));
        downsample(X, Ux, N, X0, D, N0, rx);
    }
    if (ny) {
        Y0 = Y;
        M0 = M;
        M = sz.M = ny;
        Y = static_cast<double *>(calloc((size_t)D * M, sd));
        Uy = static_cast<int *>(calloc((size_t)D * M, si));
        downsample(Y, Uy, M, Y0, D, M0, ry);
    }
    if ((nx || ny) && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr, " done. \n\n");
    // ダウンサンプリングされた各点に対応するジオデジックカーネルの値を、新しいLQ配列に適用する．
    if (ny && geok) {
        LQ0 = LQ;
        LQ = static_cast<double *>(calloc((size_t)K + (size_t)K * M, sd));
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
    wd = static_cast<double *>(calloc(dsz, sd));
    x = static_cast<double *>(calloc(xsz, sd));
    a = static_cast<double *>(calloc(M, sd));
    u = static_cast<double *>(calloc((size_t)D * M, sd));
    R = static_cast<double *>(calloc((size_t)D * D, sd));
    sgm = static_cast<double *>(calloc(M, sd));
    wi = static_cast<int *>(calloc(isz, si));
    y = static_cast<double *>(calloc(ysz, sd));
    w = static_cast<double *>(calloc(M, sd));
    v = static_cast<double *>(calloc((size_t)D * M, sd));
    t = static_cast<double *>(calloc(D, sd));
    pf = static_cast<double *>(calloc((size_t)3 * pm.nlp, sd));

    /* main computation */
    lp = bcpd(x, y, u, v, w, a, sgm, &s, R, t, &r, &Np, pf, wd, wi, X, Y, LQ, sz, pm);

    /* interpolation */

    QueryPerformanceCounter(tv + 4);
    tt[4] = clock();

    if (ny) {
        if (!(pm.opt & PW_OPT_QUIET))
            fprintf(stderr, "%s  Interpolating ... ", (pm.opt & PW_OPT_HISTO) ? "\n" : "");
        T = static_cast<double *>(calloc((size_t)D * M0, sd));
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
            x0 = static_cast<double *>(calloc((size_t)D * M0, sd));
            N0 = nx ? N0 : N;
            interpolate_x(x0, T, X0, D, M0, N0, r, pm);
            denormlize(x0, muT, sgmT, M0, D);
        }
    }

    QueryPerformanceCounter(tv + 5);
    tt[5] = clock();

    /* save interpolated variables */
    if (ny) {
        save_variable(pm.fn[OUTPUT], "y.interpolated.txt", T, D, M0, "%lf", TRANSPOSE);
        if (!(pm.opt & PW_OPT_INTPX))
            goto skip;
        save_variable(pm.fn[OUTPUT], "x.interpolated.txt", x0, D, M0, "%lf", TRANSPOSE);
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
        fprintf(stderr, "  ** Search path during optimization was saved to: [%s]\n\n", ytraj);

    free_all(x, X, wd, muX, a, v, sgm, y, Y, wi, muY, u, R, t, pf, NULL);
    // freeMesh(mesh, M);
    SetDllDirectory(NULL);

    return 0;
}
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
    // �Ƃ肠����PLY����
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
    // Mean Curvature : (> 0) = ��-plane, ( = 0) = plane or saddle, (< 0) : ��-plane
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
                     int &D,     // ������
                     int &M,     // �ʂ̐�
                     double *&Y, // �_�Q�f�[�^�i1�����z��j
                     int **&mesh // ���b�V�����
) {
    int N = V.rows(); // ���_�̐�
    D = V.cols();     // ���������i�[
    M = F.rows();     // �ʂ̐����i�[
    // �_�Q�f�[�^�̐��`
    Y = new double[D * N];
    for (int d = 0; d < D; ++d) {
        for (int n = 0; n < N; ++n) {
            Y[d + D * n] = V(n, d);
        }
    }
    // ���b�V�����̐��`
    mesh = new int *[M];
    for (int m = 0; m < M; ++m) {
        mesh[m] = new int[3];         // �O�p�`���b�V����z��
        for (int d = 0; d < 3; ++d) { // �O�p�`�̊e���_
            mesh[m][d] = F(m, d);
        }
    }
    // ���̎��_�ŁAD, M, Y, mesh�Ɋ֐��̌��ʂ��i�[����Ă���
}

void freeMesh(int **&mesh, int M) {
    for (int i = 0; i < M; ++i) {
        delete[] mesh[i];
    }
    delete[] mesh;
    mesh = nullptr;
}

int main(int argc, char **argv) {
    int d, k, l, m, n; // ���[�v�J�E���^�[
    int D, M, N, lp;   // D:�������CM:�_�QX�̓_���CN:�_�QY�̓_��
    char mode;         // �t�@�C���Ǎ����[�h
    double s, r, Np, sgmX, sgmY, *muX, *muY; // s:�X�P�[���Cr:�ό`�e���CNp:����_�̐��CsgmX,sgmY:�W���΍��C�X�P�[���̒����Ɏg�p�CmuX,muY:���σx�N�g��
    double *u, *v, *w;       // u,v:�ό`�x�N�g���Cw:�d��
    double *R, *t, *a, *sgm; // R:��]�x�N�g���Ct:���s�ړ��x�N�g���Ca:�e�_�̑Ή��m���Csgm:�e�_�̃X�P�[���ω�
    pwpm pm;                 // �A���S�̃p�����[�^���i�[����\����
    pwsz sz;                 // �T�C�Y�⎟�������i�[����\����
    double *x, *y, *X, *Y, *wd, **bX, **bY; // x,y:�ϊ���̓_�Q�CX,Y:���̓_�Q�Cwd:��Ɨp�f�[�^���i�[�CbX,bY:�t�@�C������ǂݍ��񂾐��̓_�Q
    int *wi;                                // ��Ɨp�̐����f�[�^���i�[
    int sd = sizeof(double), si = sizeof(int); // sd,si:double,int�^�̕ϐ�����������Ő�߂�T�C�Y���o�C�g�P�ʂŕێ��C��{�͂��ꂼ��8Byte,4Byte
    FILE *fp;
    char fn[256];
    int dsz, isz; // dsz,isz:double,int�^�̃f�[�^�������T�C�Y�E�A���S�ɕK�v�ȕ��������_���E�����f�[�^�̑���
    int xsz, ysz; // x,y�z��ɕK�v�ȃ������T�C�Y�C�_�Q�̎����Ɠ_�̐��A����уA���S���Y���̃I�v�V�����ɂ���ăT�C�Y���ω�
    const char *ytraj = ".optpath.bin", *xtraj = ".optpathX.bin";
    double tt[7];                       // tt:�e�����̎��Ԃ��L�^
    LARGE_INTEGER tv[7];                // ���Ԍv���p�̕ϐ�
    int nx, ny, N0, M0 = 0;             // nx,ny:�^�[�Q�b�g�E�\�[�X�̃_�E���T���v�����O���̓_���CN0,M0:���̓_�̐�
    double rx, ry, *T, *X0, *Y0 = NULL; // rx,ry:�_�E���T���v�����O�̔䗦�CT:�ϊ���̓_�Q�CX0,Y0:�_�E���T���v�����O�O�̓_�Q
    double sgmT, *muT;                  // sgmT:�ϊ���̓_�Q�̕W���΍��CmuT:�ϊ���̓_�Q�̕��σx�N�g��
    double *pf;                         // pf:�A���S�̐��\���L�^
    double *LQ = NULL, *LQ0 = NULL; // LQ,LQ0:�W�I�f�W�b�N�J�[�l�������̌��ʂ��i�[����z��
    int *Ux, *Uy; // Ux,Uy:�_�E���T���v�����O���Ɍ��_�Qindex��ۑ��Dex:�_�E���T���v�����O��̃^�[�Q�b�g�_�Q��i�Ԃ̓_�����_�Q��Ux[i]�Ԃ̓_�ɑΉ�
    int K;        // Geodesic Kernel�̌`��\���ɕK�v�Ȋ��x�N�g���̐�
    int geok = 0; // geok:�W�I�f�W�b�N�J�[�l�����g�p���邩�ǂ����̃t���O
    double *x0;   // x0:��Ԃ₻�̑��̌㏈���Ŏg�p�����C�ϊ���̓_�Q�f�[�^���i�[���邽�߂̔z��

    /* �t�@�C���̓ǂݍ��� */
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

    /* ���b�V���̓ǂݍ��� */
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
    // visualizeModel(Y_verts, Y_faces, Y_Curv.Curv);

    /* �����̏����� */
    init_genrand64(pm.rns ? pm.rns : clock());

    /* �������̊m�F */
    if (D != sz.D) {
        printf("ERROR: Dimensions of X and Y are incosistent. dim(X)=%d, dim(Y)=%d\n", sz.D, D);
        exit(EXIT_FAILURE);
    }
    if (N <= D || M <= D) {
        printf("ERROR: #points must be greater than dimension\n");
        exit(EXIT_FAILURE);
    }

    /* ���������C�A�E�g�̕ύX */
    // 1�����z��[x1, x2, x3, ..., xn, y1, y2, y3, ..., yn, z1, z2, z3, ..., zn]�ɕύX
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

    Eigen::MatrixXd YV; // 頂点座標を格納する行�E
    Eigen::MatrixXi YF; // メチE��ュの面惁E��を格納する行�E
    int **mesh;

    //// if (!igl::readPLY(pm.fn[SOURCE], YV, YF)) {
    // if (!igl::readPLY("../data/armadillo_mesh.ply", YV, YF)) {
    //    std::cerr << "Failed to read SOURCE PLY file." << std::endl;
    //    exit(EXIT_FAILURE);
    //}

    // convertToFormat(YV, YF, D, M, Y, mesh);

    printf("%d------------------------", D);
    printf("%d------------------------", M);

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

    /* �ʒu�C�X�P�[���̐��K�� */
    muX = static_cast<double *>(calloc(D, sd));
    muY = static_cast<double *>(calloc(D, sd));
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3))
        print_norm(X, Y, D, N, M, 1, pm.nrm);
    normalize_batch(X, muX, &sgmX, Y, muY, &sgmY, N, M, D, pm.nrm);
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3))
        print_norm(X, Y, D, N, M, 0, pm.nrm);

    /*Geodesic Kernel�̌v�Z */
    QueryPerformanceCounter(tv + 1);
    tt[1] = clock();
    // nnk:�ߗׂ̓_�̐����w�肷��p�����[�^�Cpm.fn[FACE_Y]:���b�V���̖ʏ����܂ރt�@�C�����Cpm.tau:Geodesic
    // Kernel�̌v�Z�Ɏg�p�����臒l
    geok = (pm.nnk || strlen(pm.fn[FACE_Y])) && pm.tau > 1e-5;
    // Fast Point Set Alignment
    if (geok && !(pm.opt & PW_OPT_QUIET))
        fprintf(stderr, "  Executing the FPSA algorithm ... ");
    if (geok) {
        sgraph *sg;
        if (pm.nnk)
            sg = sgraph_from_points(Y, D, M, pm.nnk, pm.nnr); // �_�Q����X�p�[�X�O���t���\�z
        else
            sg = sgraph_from_mesh(Y, D, M, pm.fn[FACE_Y]); // ���b�V������X�p�[�X�O���t���\�z
        // �X�p�[�X�O���t���Geodesic Kernel�������s���C���̌��ʂ��i�[����D
        // �X�p�[�X�O���t�̃G�b�W���sg->E�Əd��sg->W���g�p���A�p�����[�^�Ƃ���pm.K�i���̐��j�Apm.bet�Apm.tau�Apm.eps���󂯎��D
        // pm.bet:�ό`�̊��炩���⍄���𐧌䂷��p�����[�^�D�傫���قǁA��芊�炩�ȕό`��������A�������قǋǏ��I�ȕό`�����e�����D
        // pm.tau:�����̉e���𐧌䂷��臒l��X�P�[���p�����[�^�D�������قǁA�߂��_���m�̊֌W����������A�傫���قǉ����_���m�̊֌W���l���ɓ������D
        // pm.eps:�����v�Z�ɂ������������
        LQ = geokdecomp(&K, Y, D, M, (const int **)sg->E, (const double **)sg->W, pm.K, pm.bet, pm.tau, pm.eps);
        sz.K = pm.K = K; /* update K */
        sgraph_free(sg);
        if (geok && !(pm.opt & PW_OPT_QUIET))
            fprintf(stderr, "done. (K->%d)\n\n", K);
    }

    QueryPerformanceCounter(tv + 2);
    tt[2] = clock();

    nx = pm.dwn[TARGET];
    rx = pm.dwr[TARGET]; // �_�E���T���v�����O�ڕW�_���Ɣ䗦
    ny = pm.dwn[SOURCE];
    ry = pm.dwr[SOURCE]; // �_�E���T���v�����O�ڕW�_���Ɣ䗦
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
    // �_�E���T���v�����O���ꂽ�e�_�ɑΉ�����W�I�f�W�b�N�J�[�l���̒l���A�V����LQ�z��ɓK�p����D
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
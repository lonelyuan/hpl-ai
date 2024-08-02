#include "hpl-ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>
#include <ctype.h>

typedef struct{
    bool isPass;
    double Flops;
    double MemUsage;
}EvaPara;
//
void add(EvaPara* scoreindex, EvaPara* scoreindex1) {
    scoreindex->isPass = (scoreindex->isPass && scoreindex1->isPass);
    scoreindex->Flops += scoreindex1->Flops;
    scoreindex->MemUsage += scoreindex1->MemUsage;
}

EvaPara MxHPLTest(int n) {
    int max_it = 50;  // maximum number of iterations in GMRES
    if (max_it >= n) {
        max_it = n - 1;
    }
    // for performance&meomery statistics.
    EvaPara scoreindex;
    FILE* fp = fopen("/proc/self/status", "r");
    char line[128];
    double time_total;
    int lda = (n + 16 - 1) / 16 * 16;  // round up to multiple of 16
    unsigned long long iseed = 1;      // RNG seed
    double* A = (double*)malloc(lda * n * sizeof(double));
    double* LU = (double*)malloc(lda * n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    // generate random data.
    matgen(A, lda, n, iseed);
    vecgen(b, n, iseed+1);
// ==================================Core linear solver part====================
#pragma AutoMxPrec(start)
    time_total = get_wtime();
    copy_mat(A, lda, LU, lda, n, n);
    copy_mat(b, n, x, n, n, 1);
    // LU factorization without pivoting.
    dgetrf_nopiv(n, n, LU, lda);
    // Forward and backward substitution.
    dtrsm('L', 'L', 'N', 'U', n, 1, 1.0, LU, lda, x, n);
    dtrsm('L', 'U', 'N', 'N', n, 1, 1.0, LU, lda, x, n);
    // memory check.
    while (fgets(line, 128, fp) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            scoreindex.MemUsage = atoi(line+6);
            break;
        }
    }
    fclose(fp);
    // Using GMRES without restart.
    // GMRES is checking preconditioned residual so the tolerance is smaller.
    double tol = DBL_EPSILON / 2.0 / ((double)n / 4.0);
    gmres(n, A, lda, x, b, LU, lda, max_it, 1, tol);
    time_total = get_wtime() - time_total;
#pragma AutoMxPrec(end)
// ==================================Core linear solver part====================
    double ops = 2.0 / 3.0 * n * n * n + 3.0 / 2.0 * n * n;
    // Check final backward error.
    double norm_A = dlange('I', n, n, A, lda);
    double norm_x = dlange('I', n, 1, x, n);
    double norm_b = dlange('I', n, 1, b, n);
    dgemv('N', n, n, 1.0, A, lda, x, 1, -1.0, b, 1);
    double threshold = 1.0;
    double eps = DBL_EPSILON / 2;
    double error = dlange('I', n, 1, b, n) / (norm_A * norm_x + norm_b) / n / eps;
    scoreindex.isPass = true;
    scoreindex.Flops = 1e-9 * ops / time_total;
    if (error > threshold) {
        scoreindex.isPass = false;
        printf("n = %d: Fail!!!! scaled backward error = %5.3f > threshold = %5.3f\n", n, error, threshold);
    }
    free(A);
    free(LU);
    free(b);
    free(x);
    return scoreindex;
}


int main(int argc, char* argv[]) {
    //
    printf(
        "=============================================================================  \n");
    printf(
        "                        HPL-AI Mixed-Precision Benchmark                       \n");
    printf(
        "originally Written by Yaohung Mike Tsai, Innovative Computing Laboratory, UTK  \n");
    printf(
        "     Please visit http://www.icl.utk.edu/research/hpl-ai for more details.     \n");
    printf(
        "=============================================================================  \n");
    printf("\n");
    printf(
        "This is a revised reference implementation for LU-GMRES linear solver.         \n");
    printf(
        "It is used for the National Computer System Development Capability Competition.\n");
    printf(
        "==============Compiler System Challenge(Huawei BiSheng Cup, 2024)============  \n");
    printf(
        " Please visit https://compiler.educg.net/#/index?TYPE=COM_C for more details.  \n");
    printf("\n");
    //
    int TestNum = 1;
    int MinMatSize = 128;
    int MaxMatSize = 25600;
    if (argc == 4) {
        TestNum = atoi(argv[1]);
        MinMatSize = atoi(argv[2]);
        MaxMatSize = atoi(argv[3]);
    }
    EvaPara* scoreindex = (EvaPara*)malloc(TestNum * sizeof(EvaPara));
    EvaPara* scoreindex0 = (EvaPara*)malloc(TestNum * sizeof(EvaPara));
    //
    int PassNum = 0;
    int AveNum = 5;
    for (int i = 0; i < TestNum; ++i) {
        int n = MinMatSize + (MaxMatSize - MinMatSize) * i / (TestNum-1);
        scoreindex[i].isPass = true;
        scoreindex[i].Flops = 0.0;
        scoreindex[i].MemUsage = 0.0;
        for (int j = 0; j < AveNum; ++j) {
            scoreindex0[i] = MxHPLTest(n);
            add((scoreindex+i), (scoreindex0+i));
        }
        if (scoreindex[i].isPass) PassNum += 1;
        scoreindex[i].Flops /= AveNum;
        scoreindex[i].MemUsage /= AveNum;
    }
    printf("===============================Test Results==================================  \n");
    printf("                       (After Five-time Averaging)                             \n");
    printf("Passing Rate: %6.2f\%\n", 100*PassNum * 1.0f / TestNum);
    if (PassNum == 0) return 0;
    double* baseperf = (double*)malloc(3 * sizeof(double));
    baseperf[0] = 100000.0;
    baseperf[1] = 0.0;
    baseperf[2] = 0.0;
    for (int i = 0; i < TestNum; ++i) {
        int n = MinMatSize + (MaxMatSize - MinMatSize) * i / (TestNum-1);
        if (scoreindex[i].isPass) {
            printf("n = %d, Performance = %8.4f Gflops, MemUsage = %8.4f MB\n",
                   n, scoreindex[i].Flops, scoreindex[i].MemUsage / 1000 );
            if (baseperf[0] > scoreindex[i].Flops) {
                baseperf[0] = scoreindex[i].Flops;
            }
            if (baseperf[2] < scoreindex[i].Flops) {
                baseperf[2] = scoreindex[i].Flops;
            }
            baseperf[1] += scoreindex[i].Flops;
        }
        if (!scoreindex[i].isPass) {
            printf("n = %d, Performance = NAN, MemUsage = NAN. Failed!!!\n", n);
        }
    }
    baseperf[1] /= PassNum;
    printf("Smallest/Average/Largest Performance = %8.4f Gflops, %8.4f Gflops, %8.4f Gflops\n",
            baseperf[0], baseperf[1], baseperf[2]);
    printf("===============================Test Results==================================  \n");
    free(scoreindex0);
    free(scoreindex);
    free(baseperf);
    return 0;
}


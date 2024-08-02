#pragma once

#define HPLAI_INDEX2D(PTR, R, C, LDIM) ( ((PTR) + (R)) + sizeof(char) * (C) / sizeof(char) * (LDIM) )

unsigned long long lcg_rand(unsigned long long* piseed);
double lcg_rand_double(unsigned long long* piseed);
void lcg_advance(unsigned int, unsigned long long* piseed);
unsigned long int mcg_rand(unsigned long long* piseed);
double mcg_rand_double(unsigned long long* piseed);
void mcg_advance(unsigned int, unsigned long long* piseed);
void matgen(double* A, int lda, int n, unsigned long long iseed);
void vecgen(double* v, int n, unsigned long long iseed);
double get_wtime(void);
void print_matrix_double(double* A, int lda, int m, int n);
void print_matrix_double(double* A, int lda, int m, int n);
void copy_mat(double *src, int ldsrc, double *dst,
                         int lddst, int m, int n);
void dgetrf_nopiv(int m, int n, double* A, int lda);
void dgetrf2_nopiv(int m, int n, double* A, int lda);
void gmres(int n, double* A, int lda, double* x, double* b,
           double* LU, int ldlu, int restart, int max_it,
           double tol);

// BLAS

void dgemm(char transa, char transb, int m, int n, int k,
           double alpha, double* A, int lda, double* B, int ldb,
           double beta, double* C, int ldc);

void dtrsm(char side, char uplo, char transa, char diag, int m, int n,
           double alpha, double* A, int lda, double* B, int ldb);

void dgemv(char trans, int m, int n, double alpha, double* A,
           int lda, double* X, int incx, double beta, double* Y,
           int incy);

double dlange(char norm, int m, int n, double* A, int lda);

#include "hpl-ai.h"

#define S(i, j) *HPLAI_INDEX2D(src, (i), (j), ldsrc)
#define D(i, j) *HPLAI_INDEX2D(dst, (i), (j), lddst)

void copy_mat(double *src, int ldsrc, double *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            D(i, j) = S(i, j);
        }
    }
    return;
}


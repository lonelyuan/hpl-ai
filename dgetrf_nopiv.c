#include <stdio.h>

#include "hpl-ai.h"

#define A(i, j) *HPLAI_INDEX2D(A, (i), (j), lda)

void dgetrf_nopiv(int m, int n, double *A, int lda) {

    int j;
    int nb = 32;
    int jb = nb;

    // Use unblock code.
    if (nb > m || nb > n) {
        dgetrf2_nopiv(m, n, A, lda);
        return;
    }

    int min_mn = m < n ? m : n;

    for (j = 0; j < min_mn; j += nb) {
        if (min_mn - j < nb) {
            jb = min_mn - j;
        }

        // Factor panel
        dgetrf2_nopiv(m - j, jb, &A(j, j), lda);

        if (j + jb < n) {
            dtrsm('L', 'L', 'N', 'U', jb, n - j - jb, 1.0, &A(j, j), lda,
                  &A(j, j + jb), lda);

            if (j + jb < m) {
                dgemm('N', 'N', m - j - jb, n - j - jb, jb, -1.0, &A(j + jb, j),
                      lda, &A(j, j + jb), lda, 1.0, &A(j + jb, j + jb), lda);
            }
        }
    }
}

void dgetrf2_nopiv(int m, int n, double *A, int lda) {

    int i;

    if (m <= 1 || n == 0) {
        return;
    }

    if (n == 1) {
        for (i = 1; i < m; i++) {
            A(i, 0) /= A(0, 0);
        }
    } else {  // Use recursive code

        int n1 = (m > n ? n : m) / 2;
        int n2 = n - n1;

        dgetrf2_nopiv(m, n1, A, lda);

        dtrsm('L', 'L', 'N', 'U', n1, n2, 1.0, A, lda, &A(0, n1), lda);

        dgemm('N', 'N', m - n1, n2, n1, -1.0, &A(n1, 0), lda, &A(0, n1), lda,
              1.0, &A(n1, n1), lda);

        dgetrf2_nopiv(m - n1, n2, &A(n1, n1), lda);
    }
    return;
}

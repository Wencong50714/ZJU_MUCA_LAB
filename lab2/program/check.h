#include <stdio.h>
#include <math.h>
#include <unistd.h> // sleep, fork, getpid
#include <signal.h> // kill

/* a and b point to n x n matrices. This method checks that A = B^T. */
void checkTransposed(const float *a, const float *b, int m, int n) {
    bool correct = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i + m * j] != b[j + n * i]) {
                correct = false;
                fprintf(stderr,
                    "Transpose failed: a[%d, %d] != b[%d, %d], %f != %f\n",
                    i, j, j, i, a[i + m * j], b[j + n * i]);
                assert(correct);
            }
        }
    }

    assert(correct);
}


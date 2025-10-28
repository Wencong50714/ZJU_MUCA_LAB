/* 
 * MACA Kernels
 * Dezheng Yan, 2025
 */

#ifndef MACA_TRANSPOSE_MCH
#define MACA_TRANSPOSE_MCH

#include "common.h"

void macaTranspose(
    const float *d_input,
    float *d_output,
    int m,
    int n);

#endif

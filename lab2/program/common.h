/* 
 * MXMACA Common Header Files
 * Dezheng Yan, 2024
 */

/*
 * 
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <cassert>

#include "mc_runtime.h"

/*
 * NOTE: You can use this macro to easily check mxmaca error codes 
 * and get more information. 
 */
#define gpu_errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(mcError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != mcSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            mcGetErrorString(code), file, line);
        exit(code);
    }
}
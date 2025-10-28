/* 
 * MXMACA Kernels
 * Dezheng Yan, 2025
 */
#include <mc_runtime.h>
#include "kernel.h"

/*
 * TODO for kernel optimization:
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each wave handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * wave 0 handles block (0, 0), wave 1 handles (1, 0), wave 2 handles (0, 1),
 * wave n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304 matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void gpuTransposeKernel_V0(const float *input, float *output, int m, int n) {
    // porting from cpuTranspose()
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // 检查边界条件：i必须在[0, m-1]范围内，j必须在[0, n-1]范围内
    if (i < m) {
        for (; j < end_j; j++) {
            if (j < n) {
                output[j + n * i] = input[i + m * j];
            }
        }
    }
}

// V1 kernel: use shared memory
__global__
void gpuTransposeKernel_V1(const float *input, float *output, int m, int n) {
    __shared__ float data[64][65];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // Load data into shared memory
    for (int jj = j; jj < end_j; jj++) {
        data[threadIdx.x][jj - (64 * blockIdx.y)] = input[i + m * jj];
    }
    __syncthreads();

    for (; j < end_j; j++)
        output[j + n * i] = data[threadIdx.x][j - (64 * blockIdx.y)];
}

// V2 kernel: unroll the loops
__global__
void gpuTransposeKernel_V2(const float *input, float *output, int m, int n) {
    __shared__ float data[64][65];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // Load data into shared memory
    #pragma unroll 4
    for (int jj = j; jj < end_j; jj++) {
        data[threadIdx.x][jj - (64 * blockIdx.y)] = input[i + m * jj];
    }
    __syncthreads();

    #pragma unroll 4
    for (; j < end_j; j++)
        output[j + n * i] = data[threadIdx.x][j - (64 * blockIdx.y)];
}

// V3 kernel: negative example, show why uncoalesced access is bad
__global__
void gpuTransposeKernel_V3(const float *input, float *output, int m, int n) {
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;

    // Load data into shared memory
    float f1 = input[i + m * j];
    float f2 = input[i + m * (j + 1)];
    float f3 = input[i + m * (j + 2)];
    float f4 = input[i + m * (j + 3)];

    output[j + n * i] = f1;
    output[(j + 1) + n * i] = f2;
    output[(j + 2) + n * i] = f3;
    output[(j + 3) + n * i] = f4;
}

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define TILE_DIM 32

// V4 kernel

__global__
void gpuTransposeKernel_V4(const float *input, float *output, const int m, const int n) {
    __shared__ float data[TILE_DIM][TILE_DIM + 1];

    const int i = threadIdx.x + TILE_DIM * blockIdx.x;
    int j = 4 * threadIdx.y + TILE_DIM * blockIdx.y;
    const int end_j = j + 4;

    // Load data into shared memory
    float4 x_val;
    x_val.x = input[i + m * j];
    x_val.y = input[i + m * (j + 1)];
    x_val.z = input[i + m * (j + 2)];
    x_val.w = input[i + m * (j + 3)];
    data[threadIdx.x][4 * threadIdx.y] = x_val.x;
    data[threadIdx.x][4 * threadIdx.y + 1] = x_val.y;
    data[threadIdx.x][4 * threadIdx.y + 2] = x_val.z;
    data[threadIdx.x][4 * threadIdx.y + 3] = x_val.w;
    __syncthreads();

    float4 smem_val;
    smem_val.x = data[threadIdx.x][4 * threadIdx.y];
    smem_val.y = data[threadIdx.x][4 * threadIdx.y + 1];
    smem_val.z = data[threadIdx.x][4 * threadIdx.y + 2];
    smem_val.w = data[threadIdx.x][4 * threadIdx.y + 3];
    reinterpret_cast<float4 *>(output)[(j + n * i) / 4] = FLOAT4(smem_val);
}

void macaTranspose(
    const float *d_input,
    float *d_output,
    int m,
    int n)
{
    dim3 blockSize(TILE_DIM, TILE_DIM / 4);
    dim3 gridSize(m / TILE_DIM, n / TILE_DIM);
    gpuTransposeKernel_V4<<<gridSize, blockSize>>>(d_input, d_output, m, n);
}


// // V5 kernel
// __global__
// void gpuTransposeKernel_V5(const float *input, float *output, const int m, const int n) {
//         __shared__ float data[TILE_DIM][TILE_DIM + 1];

//     const int i = threadIdx.x + TILE_DIM * blockIdx.x;
//     int j = 8 * threadIdx.y + TILE_DIM * blockIdx.y;
//     const int end_j = j + 8;

//     // Load data into shared memory
//     float4 val1;
//     float4 val2;
//     val1.x = input[i + m * j];
//     val1.y = input[i + m * (j + 1)];
//     val1.z = input[i + m * (j + 2)];
//     val1.w = input[i + m * (j + 3)];
//     val2.x = input[i + m * (j + 4)];
//     val2.y = input[i + m * (j + 5)];
//     val2.z = input[i + m * (j + 6)];
//     val2.w = input[i + m * (j + 7)];
//     data[threadIdx.x][8 * threadIdx.y] = val1.x;
//     data[threadIdx.x][8 * threadIdx.y + 1] = val1.y;
//     data[threadIdx.x][8 * threadIdx.y + 2] = val1.z;
//     data[threadIdx.x][8 * threadIdx.y + 3] = val1.w;
//     data[threadIdx.x][8 * threadIdx.y + 4] = val2.x;
//     data[threadIdx.x][8 * threadIdx.y + 5] = val2.y;
//     data[threadIdx.x][8 * threadIdx.y + 6] = val2.z;
//     data[threadIdx.x][8 * threadIdx.y + 7] = val2.w;
//     __syncthreads();

//     float4 smem_val1;
//     float4 smem_val2;
//     smem_val1.x = data[threadIdx.x][8 * threadIdx.y];
//     smem_val1.y = data[threadIdx.x][8 * threadIdx.y + 1];
//     smem_val1.z = data[threadIdx.x][8 * threadIdx.y + 2];
//     smem_val1.w = data[threadIdx.x][8 * threadIdx.y + 3];
//     smem_val2.x = data[threadIdx.x][8 * threadIdx.y + 4];
//     smem_val2.y = data[threadIdx.x][8 * threadIdx.y + 5];
//     smem_val2.z = data[threadIdx.x][8 * threadIdx.y + 6];
//     smem_val2.w = data[threadIdx.x][8 * threadIdx.y + 7];
//     reinterpret_cast<float4 *>(output)[(j + n * i) / 4] = FLOAT4(smem_val1);
//     reinterpret_cast<float4 *>(output)[(j + n * i) / 4 + 1] = FLOAT4(smem_val2);
// }

// void macaTranspose(
//     const float *d_input,
//     float *d_output,
//     int m,
//     int n)
// {
//     dim3 blockSize(TILE_DIM, TILE_DIM / 8);
//     dim3 gridSize(m / TILE_DIM, n / TILE_DIM);
//     gpuTransposeKernel_V5<<<gridSize, blockSize>>>(d_input, d_output, m, n);
// }
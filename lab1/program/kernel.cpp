#include "kernel.h"

void call_parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {
    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 1024;

    // Initialize sum array
    mcMemset(sum, 0, (n / 1024) * sizeof(int));

    // Launch kernel
    parallel_sum_every_2nd_element<<< numberOfBlocks, threadsPerBlock >>>(sum, arr, n);
}

// parallel_sum_every_2nd_element_V6
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {

    // 0) Get thread and block information
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int lane_id = tid % 64;

    // vectorize setting
    const int4* arr_vec = (const int4*)arr;
    const int chunk_size_vec = 256;
    const int num_total_vecs = n / 4;

    // 1) Store each thread's local sum into register
    int thread_local_sum = 0;
    // (Grid-stride loop)
    #pragma unroll 4
    for (int chunk_idx = block_id;
        chunk_idx * chunk_size_vec < num_total_vecs;
        chunk_idx += gridDim.x) {
        
        #pragma unroll
        for (int i = tid; i < chunk_size_vec; i += blockDim.x) {
            int g_idx = chunk_idx * chunk_size_vec + i;
            int4 data = arr_vec[g_idx];
            thread_local_sum += data.x + data.z;
        }
    }

    // 2) Warp reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        thread_local_sum += __shfl_down_sync(0xFFFFFFFFFFFFFFFF, thread_local_sum, offset);
    }

    // 3) Block reduction
    __shared__ int warp_sums[4];
    if (lane_id == 0) {
        warp_sums[tid / 64] = thread_local_sum;
    }
    __syncthreads();

    // 4) Write output to global memory
    if (tid == 0) {
        int block_sum = 0;
        for (int i = 0; i < 4; i++) {
            block_sum += warp_sums[i];
        }
        sum[block_id] = block_sum;
    }
}
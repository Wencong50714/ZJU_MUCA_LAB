#include "kernel.h"

void call_parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {
    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = n / 1024 / 128;
    parallel_sum_every_2nd_element<<< numberOfBlocks, threadsPerBlock >>>(sum, arr, n);

}

// parallel_sum_every_2nd_element_V1 performs parallel sum without atomic.
// This code does not use any advance optimization technique on GPU,
// But still acheives many fold performance gain.
// parallel_sum_every_2nd_element_V1: 无原子的解决方案（sum变量变成一维数组）。
// 这段代码没有使用任何高级的GPU优化技术，但仍然实现了多倍的性能提升。
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 1024;
       i += blockDim.x * gridDim.x) {
    int local_sum = 0;
    for (int j = i * 1024; j < (i * 1024 + 1024); j++) {
      if(j%2==0) {
        local_sum += arr[j];
      }
    }
    sum[i] = local_sum;
  }
}

/*
// parallel_sum_every_2nd_element_V2: First read into a thread-local array, then reduce step by
step.
// By using a phased reduction approach, the operations within each for loop are
independent,
// thus enabling parallelization.
// parallel_sum_every_2nd_element_V2: 先读取到线程局部数组，然后分步缩减.
//
采用分阶段缩简的方法，这样每个for循环内部的操作都是独立的，从而可以实现并行化.
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {

        // Fill the size of local_sum
        int local_sum[];

        // First read 1024 elements into a thread-local array
        for (int j = 0; j < 1024; j++) {
            local_sum[j] = arr[i * 1024 + j];
        }

        // Process 1024 elements using a reduction algorithm with logarithmic
step size (alternating method). for (int j = 0; j < 512; j++) { local_sum[j] +=
local_sum[j + 512];
        }

        // Process 512 elements, please fill the right index
        for (int j = 0; j < 256; j++) {
            local_sum[] += local_sum[];
        }

        // Process 256 elements, please fill the right index
        for (int j = 0; j < 128; j++) {
            local_sum[] += local_sum[];
        }

        // Process 128 elements, please fill the right index
        for (int j = 0; j < 64; j++) {
            local_sum[] += local_sum[];
        }

        // Process 64 elements, please fill the right index
        for (int j = 0; j < 32; j++) {
            local_sum[] += local_sum[];
        }

        // Process 32 elements, please fill the right index
        for (int j = 0; j < 16; j++) {
            local_sum[] += local_sum[];
        }

        // Process 16 elements, please fill the right index
        for (int j = 0; j < 8; j++) {
            local_sum[] += local_sum[];
        }

        // Process 8 elements, please fill the right index
        for (int j = 0; j < 4; j++) {
            local_sum[] += local_sum[];
        }

        // Process 4 elements, please fill the right index
        for (int j = 0; j < 2; j++) {
            local_sum[] += local_sum[];
        }

        // Process 2 elements, please fill the right index
        for (int j = 0; j < 1; j++) {
            local_sum[] += local_sum[];
        }

        // Fill the index of local_sum with the sum of 1024 elements above
        sum[i] = local_sum[];
    }
}
*/

/*
// parallel_sum_every_2nd_element_V3: Uses the shared memory of the thread block.
// - Upgrades the local variable local_sum to an array shared by the thread
block,
// - Replaces the loop variable j with the thread index and i with the block
index(to achieve parallel computation).
// (Note: The result obtained by parallel_sum_every_2nd_element_V3 is not correct. Why is that?)
// parallel_sum_every_2nd_element_V3: 使用线程块的共享内存（shared memory）。
// - 把局部变量local_sum升级为线程块共享的数组，
// -
将循环变量j替换为线程的编号，将i替换为线程块的编号，以此来实现真正的并行计算。
// （注意：parallel_sum_every_2nd_element_V3得到的计算结果并不正确，这是为什么呢？）
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {

    // Fill the size of shared memory
    __shared__ int local_sum[];

    int j = ; // the thread index
    int i = ; // the block index

    // each thread read one element
    local_sum[j] = arr[i * 1024 + j];

    // using a reduction algorithm with logarithmic step size (alternating
method)
    // fill the right index on below codes
    if (j < 512) {
        local_sum[] += local_sum[];
    }
    if (j < 256) {
        local_sum[] += local_sum[];
    }
    if (j < 128) {
        local_sum[] += local_sum[];
    }
    if (j < 64) {
        local_sum[] += local_sum[];
    }
    if (j < 32) {
        local_sum[] += local_sum[];
    }
    if (j < 16) {
        local_sum[] += local_sum[];
    }
    if (j < 8) {
        local_sum[] += local_sum[];
    }
    if (j < 4) {
        local_sum[] += local_sum[];
    }
    if (j < 2) {
        local_sum[] += local_sum[];
    }
    if (j == 0) {
        sum[i] = local_sum[] + local_sum[]; //final sum of the 1024 elements
    }
}
*/

/*
// parallel_sum_every_2nd_element_V4: 
    // Add __syncthreads() in parallel_sum_every_2nd_element_V3 to make sure all threads sync up
before moving on.
    // - Ensuring that all threads reach the point where __syncthreads() is
called before proceeding further,
    // - Debug program to output correct results
    // 在parallel_sum_every_2nd_element_V3版本上，找到所有合适的位置添加上__syncthreads()，目的是：
    // - 让所有线程都运行到 __syncthreads() 所在位置以后，才能继续执行下去;
    // - 调试程序输出正确的结果
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {

}
*/

/*
    // parallel_sum_every_2nd_element_V5/V6/V7
    // Try more possible optimization points, for example:
    // - No need for __syncthreads() within a warp
    // - Avoid warp divergence
    // - Use a grid stride loop to read multiple arr elements at once
    // 尝试更多可能的优化点，例如：
    // - 线程束内不需要__syncthreads()
    // - 避免线程束产生分化
    // - 使用网格跨步循环一次读取多个 arr 元素
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n) {

}
*/
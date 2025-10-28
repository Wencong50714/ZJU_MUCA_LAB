#include <mc_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "kernel.h"
#include "timer.h"

// #define DEBUG_WRITE_TO_FILE

/*
 * Fills fill with random numbers is [0, 1]. Size is number of elements to
 * assign.
 */
void randomFill(std::vector<std::vector<float>> &matrix, int rows, int cols) {
  std::srand(std::time(nullptr));
  matrix.resize(rows);
  for (int row = 0; row < rows; ++row) {
    matrix[row].resize(cols);
    for (int col = 0; col < cols; ++col) {
      matrix[row][col] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // matrix[row][col] = row * cols + col;  // debug
    }
  }
}

/* CPU transpose, takes an n x n matrix in input and writes to output. */
void cpuTranspose(const float *input, float *output, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      output[j + n * i] = input[i + m * j];
    }
  }
}

int main(const int argc, const char **argv) {
  const bool oj_mode = argc > 1;  // read input data from args, write output
                                  // data to file

  std::string kernel = "gpu";

  //int n = 512;  // 512, 1024, 2048, 4096
  std::array<int, 4> n_options = {512, 1024, 2048, 4096};
  std::srand(std::time(nullptr));
  int n = n_options[std::rand() % 4];

  int m = n * 2;  // 1024, 2048, 4096, 8192

  std::vector<std::vector<float>> matrix;   // rows * cols, n * m
  if (oj_mode) {
    ReadMatrix(matrix);
    n = matrix.size();
    m = matrix[0].size();
  } else {
    randomFill(matrix, n, m);
  }

  if (!(n == 512 || n == 1024 || n == 2048 || n == 4096)) {
    fprintf(stderr,
            "Program only designed to run 2n*n with n=512, 1024, 2048, or 4096\n");
  }
  assert(n % 64 == 0);

  assert(kernel == "all" || kernel == "cpu" || kernel == "gpu");
  mcEvent_t start;
  mcEvent_t stop;

#define START_TIMER()                  \
  {                                    \
    gpu_errchk(mcEventCreate(&start)); \
    gpu_errchk(mcEventCreate(&stop));  \
    gpu_errchk(mcEventRecord(start));  \
  }

#define STOP_RECORD_TIMER(name)                         \
  {                                                     \
    gpu_errchk(mcEventRecord(stop));                    \
    gpu_errchk(mcEventSynchronize(stop));               \
    gpu_errchk(mcEventElapsedTime(&name, start, stop)); \
    gpu_errchk(mcEventDestroy(start));                  \
    gpu_errchk(mcEventDestroy(stop));                   \
  }

  // Initialize timers
  float cpu_ms = -1;
  float gpu_ms = -1;

  // Allocate host memory
  float *input = new float[m * n];
  float *output_cpu = new float[n * m];
  float *output_gpu = new float[n * m];
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < m; ++c) {
      input[r * m + c] = matrix[r][c];
    }
  }
#ifdef DEBUG_WRITE_TO_FILE
  if (!oj_mode) WriteFile("1-input.txt", input, n, m);
#endif

  // Allocate device memory
  float *d_input;
  float *d_output;
  gpu_errchk(mcMalloc(&d_input, m * n * sizeof(float)));
  gpu_errchk(mcMalloc(&d_output, n * m * sizeof(float)));

  // Copy input to GPU
  gpu_errchk(
      mcMemcpy(d_input, input, m * n * sizeof(float), mcMemcpyHostToDevice));

  // CPU implementation
  if (kernel == "cpu" || kernel == "all") {
    memset(output_cpu, 0, n * m * sizeof(float));
    START_TIMER();
    cpuTranspose(input, output_cpu, m, n);
    STOP_RECORD_TIMER(cpu_ms);

    checkTransposed(input, output_cpu, m, n);
    printf("Size %d*%d CPU: %f ms\n", m, n, cpu_ms);
  }

  // GPU implementation
  const int nIters = 10;  // repeat iterations
  float totalTime = 0.0;
  float minTime = std::numeric_limits<double>::max();
  if (kernel == "gpu" || kernel == "all") {
    for (int iter = 0; iter < nIters; iter++) {
      memset(output_gpu, 0, n * m * sizeof(float));
      gpu_errchk(mcMemset(d_output, 0, n * m * sizeof(float)));
      
      START_TIMER();
      macaTranspose(d_input, d_output, m, n);
      STOP_RECORD_TIMER(gpu_ms);
      totalTime += gpu_ms;
      if (minTime > gpu_ms) {
        minTime = gpu_ms;
      }
      printf("totalTime: %f; minTime: %f\n", totalTime, minTime);
    }
    gpu_errchk(mcMemcpy(output_gpu, d_output, n * m * sizeof(float),
                        mcMemcpyDeviceToHost));
    if (oj_mode) {
      WriteFile(argv[1], output_gpu, m, n);
    } else {
      checkTransposed(input, output_gpu, m, n);
#ifdef DEBUG_WRITE_TO_FILE
      WriteFile("1-output.txt", output_gpu, m, n);
#endif
    }

    float avgTime = totalTime / nIters;
    printf("avgTime: %f; minTime: %f\n", avgTime, minTime);
    printf("Size %d*%d GPU: %f ms\n", m, n, gpu_ms);
  }

  // Free host memory
  delete[] input;
  delete[] output_cpu;
  delete[] output_gpu;

  // Free device memory
  gpu_errchk(mcFree(d_input));
  gpu_errchk(mcFree(d_output));

  printf("\n");
}

#include <mc_runtime.h>

#include <cstdio>
#include <vector>

#include "check.h"
#include "helper/input.h"
#include "helper/output.h"
#include "helper_maca.h"
#include "kernel.h"
#include "maca_allocator.h"
#include "timer.h"

int main(int argc, char **argv) {
  const bool oj_mode = argc > 1;  // read input data from args, write output
                                  // data to file

  const int n = 1 << 24;  // 2^24=16777216
  const int nIters = 10;  // simulation iterations

  std::vector<int, MacaAllocator<int>> arr(n);
  std::vector<int, MacaAllocator<int>> sum(n / 1024);

  int ref_sum = 0;

  if (oj_mode) {
    // input
    std::vector<int> input_data(n);
    ReadInputData(input_data);
    for (int i = 0; i < n; i++) {
      arr[i] = input_data[i];
    }

  } else {  // local mode
    for (int i = 0; i < n; i++) {
      arr[i] = i%2+1;
      if (i%2==0) {
        ref_sum += arr[i];
      }
    }
  }

  /*******************************************************************/
  // Do not modify the code in this section.
  int output_final_sum = 0;
  double totalTime = 0.0;
  double minTime = std::numeric_limits<double>::max();
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
    /*******************************************************************/
    call_parallel_sum_every_2nd_element(sum.data(), arr.data(), n);
    checkMacaErrors(mcDeviceSynchronize());

    /*******************************************************************/
    // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
    if (minTime > tElapsed) {
      minTime = tElapsed;
    }
    /*******************************************************************/

    int final_sum = 0;
    for (int i = 0; i < n / 1024; i++) {
      final_sum += sum[i];
    }

    if (oj_mode) {
      output_final_sum = final_sum;

    } else {
      if (checkAccuracy(final_sum, ref_sum) == false) {
        printf("bad result, my final_sum[%d] != ref_sum[%d]\n", final_sum, ref_sum);
        return -1;
      }
    }
  }

  double avgTime = totalTime / (double)(nIters);

  printf("avgTime: %f; minTime: %f\n", avgTime, minTime);

  // output
  if (oj_mode) {
    OutputResult(argv[1], output_final_sum);
  }

  return 0;
}

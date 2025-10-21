#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void OutputResult(const char *file_name, int result) {
  // write result to `file_name`
  auto *OFD = fopen(file_name, "w");
  if (OFD == nullptr) {
    fprintf(stderr, "Error open result.txt by %s\n", strerror(errno));
    std::abort();
  }
  fprintf(OFD, "%d ", result);
  fprintf(OFD, "\n");
}
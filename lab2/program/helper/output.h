#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void WriteFile(const char *file_name, float *matrix, int rows,
                      int cols) {
    if (file_name == NULL || matrix == NULL || rows <= 0 || cols <= 0) {
        fprintf(stderr, "错误: 无效的参数\n");
        std::abort();
    }

    FILE *file = fopen(file_name, "wb");
    if (file == NULL) {
        fprintf(stderr, "错误: 无法打开文件 %s\n", file_name);
        std::abort();
    }

    // 写入矩阵维度
    if (fwrite(&rows, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "错误: 写入行数失败\n");
        fclose(file);
        std::abort();
    }

    if (fwrite(&cols, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "错误: 写入列数失败\n");
        fclose(file);
        std::abort();
    }

    // 写入矩阵数据
    int total_elements = rows * cols;
    if (fwrite(matrix, sizeof(float), total_elements, file) != total_elements) {
        fprintf(stderr, "错误: 写入矩阵数据失败\n");
        fclose(file);
        std::abort();
    }

    fclose(file);
}

inline void WriteTextFile(const char *file_name, float *matrix, int rows,
                      int cols) {
  // write result to `file_name`
  auto *OFD = fopen(file_name, "w");
  if (OFD == nullptr) {
    fprintf(stderr, "Error open result.txt by %s\n", strerror(errno));
    std::abort();
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      fprintf(OFD, "%f ", matrix[r * cols + c]);
    }
    fprintf(OFD, "\n");
  }
  fprintf(OFD, "\n");
}

#define OutputResult WriteFile

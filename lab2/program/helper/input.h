#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

void ReadMatrix(std::vector<std::vector<float>>& matrix) {
    std::istreambuf_iterator<char> it(std::cin);
    std::istreambuf_iterator<char> end;
    std::vector<unsigned char> buffer(it, end);
    auto p = buffer.data();

    float* p_matrix;
    int rows;
    int cols;
    memcpy(&rows, p, sizeof(int));
    p += sizeof(int);
    memcpy(&cols, p, sizeof(int));
    p += sizeof(int);

    const float* data_start = reinterpret_cast<const float*>(p);
    matrix.resize(rows);
    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = data_start[i * cols + j];
        }
    }
}

void ReadTextMatrix(std::vector<std::vector<float>>& matrix) {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<float> row;
    std::istringstream iss(line);
    float tmp;
    while (iss >> tmp) {
      row.push_back(tmp);
    }

    matrix.push_back(row);
  }
}

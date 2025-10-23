#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

inline void ReadInputData(std::vector<int>& out_data) {
  std::vector<int> data;
  MetaXOJ::DataLoader::loadData(data);

  for (int i = 0; i < out_data.size(); ++i) {
    if (i < data.size()) {
      out_data[i] = data[i];
    } else {
      out_data[i] = 0;
    }
  }
}
#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
namespace MetaXOJ {
class DataLoader {
 public:
  /*
   * load data like this :
   *
   * 1 2 3 4 5 6
   */
  template <typename Ty>
  static void loadData(std::vector<Ty> &data) {
    std::istream &input = std::cin;
    std::string line;
    if (std::getline(input, line)) {
      std::stringstream ss(line);
      Ty num;
      while (ss >> num) data.push_back(num);
    }
  }
  /*
   * load data like this :
   *
   * 1 2 3 4 5 6
   */
  template <typename Ty>
  static void loadData(std::vector<Ty> &data, const std::string file) {
    std::ifstream input(file);
    if (!input) {
      fprintf(stderr, "Error open %s by %s\n", file.c_str(), strerror(errno));
      std::abort();
    }
    std::string line;
    if (std::getline(input, line)) {
      std::stringstream ss(line);
      Ty num;
      while (ss >> num) data.push_back(num);
    }
  }
  /*
   * load data like this :
   *
   * 1 2 3 4
   * 4 5 6 4
   * 1
   * 2 3 4 9
   */
  template <typename Ty>
  static void loadData(std::vector<std::vector<Ty>> &datas) {
    std::istream &input = std::cin;
    std::string line;
    while (std::getline(input, line)) {
      std::stringstream ss(line);
      std::vector<Ty> data;
      Ty num;
      while (ss >> num) data.push_back(num);
      datas.push_back(std::move(data));
    }
  }
  /*
   * load data like this :
   *
   * 1 2 3 4
   * 4 5 6 4
   * 1
   * 2 3 4 9
   */
  template <typename Ty>
  static void loadData(std::vector<std::vector<Ty>> &datas,
                       const std::string file) {
    std::ifstream input(file);
    if (!input) {
      fprintf(stderr, "Error open %s by %s\n", file.c_str(), strerror(errno));
      std::abort();
    }
    std::string line;
    while (std::getline(input, line)) {
      std::stringstream ss(line);
      std::vector<Ty> data;
      Ty num;
      while (ss >> num) data.push_back(num);
      datas.push_back(std::move(data));
    }
  }
  /*
   * load data like this :
   *
   * hello
   * world
   * xxstring
   */
  static void loadData(std::vector<std::string> &datas) {
    std::istream &input = std::cin;
    std::string line;
    while (std::getline(input, line)) {
      datas.push_back(line);
    }
  }
  /*
   * load data like this :
   *
   * hello
   * world
   * xxstring
   */
  static void loadData(std::vector<std::string> &datas,
                       const std::string file) {
    std::ifstream input(file);
    if (!input) {
      fprintf(stderr, "Error open %s by %s\n", file.c_str(), strerror(errno));
      std::abort();
    }
    std::string line;
    while (std::getline(input, line)) {
      datas.push_back(line);
    }
  }
};

class Differ {
  std::vector<const char *> Argv;

 public:
  Differ() = default;
  void configArgs(int argc, const char *argv[]) {
    Argv.clear();
    for (int i = 0; i < argc; ++i) {
      if (argv[i] == nullptr) {
        fprintf(stderr, "Warning argv[%i] is nullptr\n", i);
        continue;
      }
      Argv.push_back(argv[i]);
    }
  }
  /*
   *diff data like this :
   *
   *1 2 3 4 5 6
   */
  template <typename Ty>
  void diff_vector() {
    if (Argv.size() < 3) {
      fprintf(stderr,
              "Error incorrect comand line format.Use <exe> "
              "<resultfile> <goldenfile>\n");
      std::abort();
    }
    std::vector<Ty> result;
    std::vector<Ty> golden;
    DataLoader::loadData(result, Argv[1]);
    DataLoader::loadData(golden, Argv[2]);
    if (result.size() != golden.size()) {
      fprintf(stderr, "Error result.size & golden.size not matched\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < result.size(); ++i) {
      if (!limit_same(result[i], golden[i])) {
        std::cerr << "Error incorrect answer your(" << result[i] << " & gloden("
                  << golden[i] << ") in col = " << i + 1 << ")\n";
      }
    }
  }
  /*
   *diff data like this :
   *
   *hello
   *world
   *xxstring
   */
  void diff_vstring() {
    if (Argv.size() < 3) {
      fprintf(stderr,
              "Error incorrect comand line format.Use <exe> "
              "<resultfile> <goldenfile>\n");
      std::abort();
    }
    std::vector<std::string> result;
    std::vector<std::string> golden;
    DataLoader::loadData(result, Argv[1]);
    DataLoader::loadData(golden, Argv[2]);
    if (result.size() != golden.size()) {
      fprintf(stderr, "Error result.size & golden.size not matched\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < result.size(); ++i) {
      if (!limit_same(result[i], golden[i])) {
        std::cerr << "Error incorrect answer your(" << result[i] << " & gloden("
                  << golden[i] << ") in row = " << i + 1 << "\n";
        exit(EXIT_FAILURE);
      }
    }
  }
  /*
   *diff data like this :
   *
   *1 2 3 4
   *4 5 6 4
   *1
   *2 3 4 9
   */
  template <typename Ty>
  void diff_matrix() {
    if (Argv.size() < 3) {
      fprintf(stderr,
              "Error incorrect comand line format.Use <exe> "
              "<resultfile> <goldenfile>\n");
      std::abort();
    }
    std::vector<std::vector<Ty>> result;
    std::vector<std::vector<Ty>> golden;
    DataLoader::loadData(result, Argv[1]);
    DataLoader::loadData(golden, Argv[2]);
    if (result.size() != golden.size()) {
      fprintf(stderr, "Error result.size & golden.size not matched\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < result.size(); ++i) {
      if (result[i].size() != golden[i].size()) {
        fprintf(stderr,
                "Error result.size & golden.size not matched in row = %d\n",
                i + 1);
        exit(EXIT_FAILURE);
      }
      for (int j = 0; j < result[i].size(); ++j) {
        if (!limit_same(result[i][j], golden[i][j])) {
          std::cerr << "Error incorrect answer your(" << result[i][j]
                    << " & gloden(" << golden[i][j] << ") in row = " << i + 1
                    << ", col = " << j + 1 << ")\n";
          exit(EXIT_FAILURE);
        }
      }
    }
  }

 private:
  template <typename Ty>
  bool limit_same(const Ty &s1, const Ty &s2) {
    return s1 == s2;
  }
  bool limit_same(float s1, float s2) { return std::abs(s1 - s2) <= 1e-5; }
  bool limit_same(double s1, double s2) { return std::abs(s1 - s2) <= 1e-8; }
};
}  // namespace MetaXOJ

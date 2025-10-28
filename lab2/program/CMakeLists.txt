# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
# set(EXECUTABLE_OUTPUT_PATH ../)

set(TARGET program)

# Set the project name
project(main)
  
# Set the path to the compiler
set(MXCC_PATH /opt/maca)
set(CMAKE_CXX_COMPILER /opt/maca/mxgpu_llvm/bin/mxcc)

# Set the compiler flags
set(MXCC_COMPILE_FLAGS -x maca)
add_compile_options(${MXCC_COMPILE_FLAGS})

# Add source files
File(GLOB SRCS main.cpp kernel.cpp)
add_executable(${TARGET} ${SRCS})

target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CPP>:--extended-lambda>)
target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CPP>:--expt-relaxed-constexpr>)
target_link_libraries(${TARGET} PRIVATE mcToolsExt)

install(TARGETS ${TARGET} DESTINATION bin)
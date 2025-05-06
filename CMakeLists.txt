cmake_minimum_required(VERSION 3.10)
project(ConvLayoutStudy)

set(CMAKE_C_FLAGS "-O3")

set(LIB_ROOT "/lib_path")

# openmp
find_package(OpenMP REQUIRED)
# openblas
set(OPENBLAS_INCLUDE_DIR "${LIB_ROOT}/openblas/include")
set(OPENBLAS_LIB_DIR "${LIB_ROOT}/openblas/lib")

# conv_layers
add_executable(conv_layers conv_layers.c)
target_include_directories(conv_layers PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(conv_layers PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(conv_layers -l:libopenblas.a OpenMP::OpenMP_C -lpthread -lm)

# nchw
add_executable(im2col_nchw im2col_nchw.c)
target_include_directories(im2col_nchw PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(im2col_nchw PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(im2col_nchw -l:libopenblas.a OpenMP::OpenMP_C -lpthread -lm)

# nhwc
add_executable(im2col_nhwc im2col_nhwc.c)
target_include_directories(im2col_nhwc PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(im2col_nhwc PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(im2col_nhwc -l:libopenblas.a OpenMP::OpenMP_C -lpthread -lm)

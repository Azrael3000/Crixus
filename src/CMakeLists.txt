
add_library(
  Core
  crixus.cpp
)

cuda_add_library(
  cuCore
  crixus.cu
  crixus_d.cu
  ini/ini.c
  ini/cpp/INIReader.cpp
)

#ifndef CRIXUS_D_CUH
#define CRIXUS_D_CUH

#include "cuda_local.cuh"
#include "lock.cuh"

typedef union{
  float4 vec;
  float a[4];
} uf4;

typedef union{
  int4 vec;
  int a[4];
} ui4;

__global__ void set_bound_elem (uf4*, uf4*, float*, ui4*, unsigned int, float*, float*, float*, float*, Lock, int);

inline __host__ __device__ float get(float4 a, const int i){
  switch(i){
    case 0:
      return a.x;
    case 1:
      return a.y;
    case 2:
      return a.z;
    case 3:
      return a.w;
  }
  return 0.;
};

#endif

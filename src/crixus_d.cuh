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

__global__ void swap_normals (uf4*, int);

__global__ void periodicity_links (uf4*, ui4*, int, int, uf4*, uf4*, float, int*, int*, int*, int);

__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, uf4*, uf4*, int*, int*, int, int, float, float, bool*);

__global__ void fill_fluid (uf4*, float, float, float, float, float, float, float, float, int*, int, Lock);

__device__ void gpu_sync (int*, int*);

#endif

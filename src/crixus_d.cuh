#ifndef CRIXUS_D_CUH
#define CRIXUS_D_CUH

#include "cuda_local.cuh"
#include "crixus.h"
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

__global__ void periodicity_links (uf4*, ui4*, int, int, uf4*, uf4*, float, int*, int*, int*, int, Lock);

#ifndef bdebug
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, uf4*, uf4*, int*, int*, int, int, float, float, bool*, Lock);
#else
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, uf4*, uf4*, int*, int*, int, int, float, float, bool*, Lock, uf4*, float*);
#endif

__global__ void calc_ngridp (uf4*, unsigned int*, uf4*, uf4*, bool*, int*, int, float, float, int, int, float, Lock, int);

__global__ void init_gpoints (uf4*, ui4*, float*, uf4*, uf4*, float*, float*, uf4*, uf4*, bool*, int, float, float, int, float, int, int, float, float, int*, Lock, float*);

__global__ void fill_fluid (uf4*, float, float, float, float, float, float, float, float, int*, int, Lock);

__device__ void gpu_sync (int*, int*);

#endif

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

__global__ void find_links(uf4 *pos, int nvert, uf4 *dmax, uf4 *dmin, float dr, int *newlink, int idim);

__global__ void periodicity_links (uf4*, ui4*, int, int, uf4*, uf4*, float, int*, int);

__global__ void calc_trisize(ui4 *ep, int *trisize, int nbe);

#ifndef bdebug
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, uf4*, uf4*,  int, int, float, float, bool*);
#else
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, uf4*, uf4*,  int, int, float, float, bool*, uf4*, float*);
#endif

__global__ void calc_ngridp (uf4*, unsigned int*, uf4*, uf4*, bool*, int*, int, float, float, int, int, float, Lock, int);

__device__ int calc_ggam(uf4 tpos, uf4 *pos, ui4 *ep, float *surf, uf4 *norm, uf4 *gpos, float *ggam, int *iggam, uf4 *dmin, uf4 *dmax, bool *per, int ngridp, float dr, float hdr, int iker, float eps, int nvert, int nbe, float krad, float iseed, bool ongpoint, int id);

__global__ void init_gpoints (uf4*, ui4*, float*, uf4*, uf4*, float*, float*, int*, uf4*, uf4*, bool*, int, float, float, int, float, int, int, float, float, int*, Lock, float*, int *);

__global__ void fill_fluid (uf4*, float, float, float, float, float, float, float, float, int*, int, Lock);

__device__ void gpu_sync (int*, int*);

#endif

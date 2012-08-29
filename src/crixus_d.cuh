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

__global__ void fill_fluid (unsigned int *fpos, unsigned int *nfi, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, uf4 *min, uf4 *dmax, float eps, float dr, Lock lock);

__device__ void gpu_sync (int*, int*);

__global__ void fill_fluid_complex (unsigned int *fpos, unsigned int *nfi, uf4 *norm, ui4 *ep, float *dist, ui4 *ind, uf4 *pos, int nbe, uf4 *dmin, uf4 *dmax, float eps, float dr, int sIndex, unsigned int sBit, Lock lock);

__device__ bool checkCollision(int si, int sj, int sk, int ei, int ej, int ek, uf4 *norm, ui4 *ep, float *dist, ui4 *ind, uf4 *pos, int nbe, float dr, uf4* dmin, ui4 dimg, float eps);

__device__ bool checkTriangleCollision(uf4 s, uf4 e, uf4 n, float d, ui4 ind, uf4 *v, float eps);

__global__ void perpareTriangles(uf4 *norm, uf4 *pos, ui4 *ep, ui4 *ind, float *dist, unsigned int nbe);

__global__ void identifyInOutFlowSegments (uf4 *pos, int nvert, int nbe, uf4 *outpos, ui4 *outep, int outnbe, uf4 *inpos, ui4 *inep, int innbe, float eps, short *inout);

__device__ bool segInTri(uf4 *vb, uf4 spos, float eps);
#endif

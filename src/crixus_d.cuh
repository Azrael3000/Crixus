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

__global__ void init_cell_idx (int *cell_idx, int *cur_cell_idx, ui4 gridn);

__global__ void count_cell_bes (uf4 *pre_pos, int *cell_idx, unsigned int nbe, unsigned int nvert, ui4 gridn, uf4 *dmin, uf4 *dmax, float eps);

__global__ void add_up_indices (int *cell_idx, int *cur_cell_idx, ui4 gridn);

__global__ void sort_bes (uf4 *pre_pos, uf4 *pos, uf4 *pre_norm, uf4 *norm, ui4 *pre_ep, ui4 *ep, float *pre_surf, float *surf, int *cell_idx, int *cur_cell_idx, unsigned int nbe, unsigned int nvert, ui4 gridn, uf4 *dmin, uf4 *dmax, float eps);

__global__ void set_bound_elem (uf4*, uf4*, float*, ui4*, unsigned int, float*, float*, float*, float*, Lock, int);

__global__ void swap_normals (uf4*, int);

__global__ void find_links(uf4 *pos, int nvert, uf4 *dmax, uf4 *dmin, float dr, int *newlink, int idim);

__global__ void periodicity_links (uf4*, ui4*, int, int, uf4*, uf4*, float, int*, int);

__global__ void calc_trisize(ui4 *ep, int *trisize, int nbe);

#ifndef bdebug
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, int*, ui4, uf4*, uf4*,  int, int, float, float, bool*);
#else
__global__ void calc_vert_volume (uf4*, uf4*, ui4*, float*, int*, int*, ui4, uf4*, uf4*,  int, int, float, float, bool*, uf4*, float*);
#endif

__global__ void fill_fluid (unsigned int *fpos, unsigned int *nfi, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, uf4 *min, uf4 *dmax, float eps, float dr, Lock lock);

__device__ void gpu_sync (int*, int*);

__global__ void fill_fluid_complex (unsigned int *fpos, unsigned int *nfi, uf4 *norm, ui4 *ep, uf4 *pos, int nbe, uf4 *dmin, uf4 *dmax, float eps, float dr, int sInd, Lock lock, int cnbe, float dr_wall, int iteration, int *cell_idx, ui4 gridn, bool *per);

__device__ bool checkCollision(int si, int sj, int sk, int ei, int ej, int ek, uf4 *norm, ui4 *ep, uf4 *pos, int nbe, float dr, uf4* dmin, ui4 dimg, float eps, int cnbe, float dr_wall, int *cell_idx, ui4 gridn, uf4 griddr, bool *per);

__device__ bool checkTriangleCollision(uf4 s, uf4 e, uf4 n, uf4 *v, float eps);

__global__ void identifySpecialBoundarySegments (uf4 *pos, ui4 *ep, int nvert, int nbe, uf4 *sbpos, ui4 *sbep, int sbnbe, float eps, int *sbid, int i);

__global__ void identifySpecialBoundaryVertices (int *sbid, int i, int *trisize, int nvert);

__global__ void checkForSingularSegments (uf4 *pos, ui4 *ep, uf4 *norm, float *surf, int nvert, int nbe, int *sbid, int sbi, float dr, float eps, bool *per, uf4 *dmin, uf4 *dmax, bool *needsUpdate);

__device__ bool segInTri(uf4 *vb, uf4 spos, uf4 norm, float eps);
#endif

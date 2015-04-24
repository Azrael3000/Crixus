#ifndef CRIXUS_H
#define CRIXUS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include "cuda_local.cuh"
#include "ini/cpp/INIReader.h"

//definitions
#define DATASETNAME "Compound"
#define sgn(x) (float)((x>0.)-(x<0.))
#define sqr(x) ((x)*(x))

//variables
const unsigned int           gres = 20;    // grid resolution = dr/dr_grid
const unsigned int         trimax = 50;    // maximum amount of triangles associated to one vertex particle
const unsigned int        maxlink = 500;   // maximum number of links (grid points to boundary elements & vertex particles
const unsigned int        ipoints = 17;    // number of unique integration points for gauss quadrature (without permutations)
const unsigned int max_iterations = 10000; // maximum number of iterations during complex filling

//Output structures
struct OutBuf{
  float x,y,z,nx,ny,nz,vol,surf;
  int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};

//function headers
int hdf5_output (OutBuf *buf, int len, const char *filename);

int vtk_output (OutBuf *buf, int len, const char *filename);

inline void scalar_array(FILE *fid, const char *type, const char *name, size_t offset);

inline void vector_array(FILE *fid, const char *type, const char *name, uint dim, size_t offset);

inline void vector_array(FILE *fid, const char *type, uint dim, size_t offset);

int crixus_main(int, char**);

/* Endianness check: (char*)&endian_int reads the first byte of the int,
 * which is 0 on big-endian machines, and 1 in little-endian machines */
static int endian_int=1;
static const char* endianness[2] = { "BigEndian", "LittleEndian" };

//debug
//#define bdebug 2344-960

#endif

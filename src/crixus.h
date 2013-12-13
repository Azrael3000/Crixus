#ifndef CRIXUS_H
#define CRIXUS_H

//definitions
#define DATASETNAME "Compound"
#define sgn(x) (float)((x>0.)-(x<0.))
#define sqr(x) ((x)*(x))

//variables
const unsigned int        maxfbox = 100; // maximum number of fluid boxes
const unsigned int           gres = 100; // grid resolution = dr/dr_grid
const unsigned int         trimax = 50;  // maximum amount of triangles associated to one vertex particle
const unsigned int        maxlink = 500; // maximum number of links (grid points to boundary elements & vertex particles
const unsigned int        ipoints = 17;  // number of unique integration points for gauss quadrature (without permutations)
const unsigned int max_iterations = 100; // maximum number of iterations during complex filling

//Output structures
struct OutBuf{
  float x,y,z,nx,ny,nz,vol,surf;
  int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};
struct gOutBuf{
  float x, y, z, gam, ggamx, ggamy, ggamz;
  int id;
};
struct linkOutBuf{
  int id, iggam;
  float ggam;
};

//function headers
int hdf5_output (OutBuf *buf, int len, const char *filename, float *timevalue);

int hdf5_grid_output (gOutBuf *buf, int len, const char *filename, float *timevalue);

int hdf5_link_output (linkOutBuf *buf, int len, const char *filename, float *timevalue);

int crixus_main(int, char**);

//debug
//#define bdebug 2344-960

#endif

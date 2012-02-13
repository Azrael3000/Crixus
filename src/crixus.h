#ifndef CRIXUS_H
#define CRIXUS_H

//definitions
#define DATASETNAME "Compound"
#define sgn(x) (float)((x>0.)-(x<0.))
#define sqr(x) ((x)*(x))

//variables
const unsigned int maxfbox = 100;// maximum number of fluid boxes
const unsigned int gres    = 10; // grid resolution = dr/dr_grid
const unsigned int trimax  = 100;// maximum amount of triangles associated to one vertex particle
const unsigned int maxlink = 500;// maximum number of links (grid points to boundary elements & vertex particles
const unsigned int ipoints = 100; // number of monte carlo integration points

//Output structures
struct OutBuf{
	float x,y,z,nx,ny,nz,vol,surf;
	int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};
struct gOutBuf{
	float x, y, z, gam;
	int id;
};
struct linkOutBuf{
	int id;
	float ggamx, ggamy, ggamz;
};

//function headers
int hdf5_output (OutBuf *buf, int len, const char *filename, float *timevalue);

int hdf5_grid_output (gOutBuf *buf, int len, const char *filename, float *timevalue);

int hdf5_link_output (linkOutBuf *buf, int len, const char *filename, float *timevalue);

int crixus_main(int, char**);

//debug
//#define bdebug 232

#endif

#ifndef CRIXUS_H
#define CRIXUS_H

//definitions
#define DATASETNAME "Compound"
#define sgn(x) (float)((x>0.)-(x<0.))

//variables
const unsigned int maxfbox = 100;// maximum number of fluid boxes
const unsigned int gres    = 10; // grid resolution = dr/dr_grid
const unsigned int trimax  = 100;

//Output struct
struct OutBuf{
	float x,y,z,nx,ny,nz,vol,surf;
	int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};

//function headers
int hdf5_output (OutBuf *buf, int len, const char *filename, float *timevalue);

int crixus_main(int, char**);

#endif

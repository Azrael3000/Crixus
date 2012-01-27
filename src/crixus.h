#define DATASETNAME "Compound"
#define sgn(x) (float)((x>0.)-(x<0.))

struct OutBuf{
	float x,y,z,nx,ny,nz,vol,surf;
	int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};

int hdf5_output (OutBuf *buf, int len, const char *filename, float *timevalue);

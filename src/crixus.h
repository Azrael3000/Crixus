#define DATASETNAME "Compound"
#define sgn(x) (double)((x>0.)-(x<0.))

struct OutBuf{
	double x,y,z,nx,ny,nz,vol,surf;
	int kpar, kfluid, kent, kparmob, iref, ep1, ep2, ep3;
};

int hdf5_output (OutBuf *buf, int len, const char *filename, double *timevalue);

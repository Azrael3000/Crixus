#ifndef CRIXUS_D_CU
#define CRIXUS_D_CU

#include <cuda.h>
#include "lock.cuh"
#include "crixus_d.cuh"

__global__ void set_bound_elem (uf4 *pos, uf4 *norm, float *surf, ui4 *ep, unsigned int nbe, float *xminp, float *xminn, float *nminp, float*nminn, Lock lock, int nvert)
{
	float ddum[3];
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ float xminp_c[threadsPerBlock];
	__shared__ float xminn_c[threadsPerBlock];
	__shared__ float nminp_c[threadsPerBlock];
	__shared__ float nminn_c[threadsPerBlock];
	float xminp_t;
	float xminn_t;
	float nminp_t;
	float nminn_t;
	int i_c = threadIdx.x;
	xminp_t = *xminp;
	xminn_t = *xminn;
	nminp_t = *nminp;
	nminn_t = *nminn;
	while(i<nbe){
		//formula: a = 1/4 sqrt(4*a^2*b^2-(a^2+b^2-c^2)^2)
		float a2 = 0.;
		float b2 = 0.;
		float c2 = 0.;
		ddum[0] = 0.;
		ddum[1] = 0.;
		ddum[2] = 0.;
		for(unsigned int j=0; j<3; j++){
			ddum[j] += pos[ep[i].a[0]].a[j]/3.;
			ddum[j] += pos[ep[i].a[1]].a[j]/3.;
			ddum[j] += pos[ep[i].a[2]].a[j]/3.;
			a2 += pow(pos[ep[i].a[0]].a[j]-pos[ep[i].a[1]].a[j],2);
			b2 += pow(pos[ep[i].a[1]].a[j]-pos[ep[i].a[2]].a[j],2);
			c2 += pow(pos[ep[i].a[2]].a[j]-pos[ep[i].a[0]].a[j],2);
		}
		if(norm[i].a[2] > 1e-5 && xminp_t > ddum[2]){
			xminp_t = ddum[2];
			nminp_t = norm[i].a[2];
		}
		if(norm[i].a[2] < -1e-5 && xminn_t > ddum[2]){
			xminn_t = ddum[2];
			nminn_t = norm[i].a[2];
		}
		surf[i] = 0.25*sqrt(4.*a2*b2-pow(a2+b2-c2,2));
    for(int j=0; j<3; j++)
		  pos[i+nvert].a[j] = ddum[j];
		i += blockDim.x*gridDim.x;
	}

	xminp_c[i_c] = xminp_t;
	xminn_c[i_c] = xminn_t;
	nminp_c[i_c] = nminp_t;
	nminn_c[i_c] = nminn_t;
	__syncthreads();

	int j = blockDim.x/2;
	while (j!=0){
		if(i_c < j){
			if(xminp_c[i_c+j] < xminp_c[i_c]){
				xminp_c[i_c] = xminp_c[i_c+j];
				nminp_c[i_c] = nminp_c[i_c+j];
			}
			if(xminn_c[i_c+j] < xminn_c[i_c]){
				xminn_c[i_c] = xminn_c[i_c+j];
				nminn_c[i_c] = nminn_c[i_c+j];
			}
		}
		__syncthreads();
		j /= 2;
	}

	if(i_c == 0){
		lock.lock();
		if(xminp_c[0] < *xminp){
			*xminp = xminp_c[0];
			*nminp = nminp_c[0];
		}
		if(xminn_c[0] < *xminn){
			*xminn = xminn_c[0];
			*nminn = nminn_c[0];
		}
		lock.unlock();
	}
}


#endif

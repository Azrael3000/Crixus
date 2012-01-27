#include "crixus_d.cuh"
#include "lock.cuh"

__global__ void set_bound_elem
	(float4 *pos, float4 *norm, float *surf, int4 *ep, unsigned int nbe,
	 float *xminp, float *xminn, float *nminp, float*nminn, Lock lock)
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
	xminp_t = xminp;
	xminn_t = xminn;
	nminp_t = nminp;
	nminn_t = nminn;
	while(i<nbe){
		//formula: a = 1/4 sqrt(4*a^2*b^2-(a^2+b^2-c^2)^2)
		float a2 = 0.;
		float b2 = 0.;
		float c2 = 0.;
		dduma[0] = 0.;
		dduma[1] = 0.;
		dduma[2] = 0.;
		for(unsigned int j=0; j<3; j++){
			dduma[j] += posa[ep[i][0]][j]/3.;
			dduma[j] += posa[ep[i][1]][j]/3.;
			dduma[j] += posa[ep[i][2]][j]/3.;
			a2 += pow(posa[ep[i][0]][j]-posa[ep[i][1]][j],2);
			b2 += pow(posa[ep[i][1]][j]-posa[ep[i][2]][j],2);
			c2 += pow(posa[ep[i][2]][j]-posa[ep[i][0]][j],2);
		}
		if(norm[i][2] > 1e-5 && xminp_t > dduma[2]){
			xminp_t = dduma[2];
			nminp_t = norma[i][2];
		}
		if(norm[i][2] < -1e-5 && xminn_t > dduma[2]){
			xminn_t = dduma[2];
			nminn_t = norma[i][2];
		}
		surf[i] = 0.25*sqrt(4.*a2*b2-pow(a2+b2-c2,2));
		posa[i+nvert][0] = dduma[0];
		posa[i+nvert][1] = dduma[1];
		posa[i+nvert][2] = dduma[2];
		i += blockDim.x*gridDim.x;
	}

	xminp_c[i_c] = xminp_t;
	xminn_c[i_c] = xminn_t;
	nminp_c[i_c] = nminp_t;
	nminn_c[i_c] = nminn_t;
	__syncthreads();

	int j = blockDim.x/2;
	while (i!=0){
		if(i_c < j){
			if(xminp_c[i_c+i] < xminp_c[i_c]){
				xminp_c[i_c] = xminp_c[i_c+i];
				nminp_c[i_c] = nminp_c[i_c+i];
			}
			if(xminn_c[i_c+i] < xminn_c[i_c]){
				xminn_c[i_c] = xminn_c[i_c+i];
				nminn_c[i_c] = nminn_c[i_c+i];
			}
		}
		__syncthreads();
		j /= 2;
	}

	if(i_c == 0){
		lock.lock();
		if(xminp_c[0] < xminp){
			xminp = xminp_c[0];
			nminp = nminp_c[0];
		}
		if(xminn_c[0] < xminn){
			xminn = xminn_c[0];
			nminn = nminn_c[0];
		}
		lock.unlock();
	}
}

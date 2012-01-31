#ifndef LOCK_CUH
#define LOCK_CUH

#include "cuda_local.cuh"

struct Lock {
		//variables
		int *mutex;

		//functions
		Lock (void){
			int state=0;
			CUDA_SAFE_CALL( cudaMalloc((void **) &mutex, sizeof(int)) );
			CUDA_SAFE_CALL( cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice) );
		}
	
		~Lock (void){
			cudaFree(mutex);
		}
	
		__device__ void lock (void){
			while(atomicCAS(mutex, 0, 1) != 0);
		}
	
		__device__ void unlock (void){
			atomicExch(mutex,0);
		}
};

#endif

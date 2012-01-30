#ifndef CUDA_LOCAL_CUH
#define CUDA_LOCAL_CUH

#include <stdio.h>
#include <cuda.h>

#define threadsPerBlock 512

#define CUDA_SAFE_CALL_NO_SYNC( call ) {                                     \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define CUDA_SAFE_CALL( call ) {                                             \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaDeviceSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#endif

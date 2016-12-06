#include "mex.h"

#include <cuda_runtime.h>                                                       
#include <helper_cuda.h>                                                        
#include <helper_functions.h>

inline int BLK(int number, int blksize)                                         
{
	return (number + blksize - 1) / blksize;
}


__global__ void kernel_vectoradd(const float* __restrict__ d_A, 
		const int N,
		float *d_B) 
{
	uint gx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	if(gx < N)
		d_B[gx] = d_A[gx] + 1.f;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// to check the arguments


	//-----------------------------------------------------------------------//
	// read input 0
	mxArray const *src 		= prhs[0];
	float *A			= (float*)mxGetData( src );

	//mwSize const *srcSize 	= mxGetDimensions( src);
	//int A_len = (srcSize[0] == 1 ? srcSize[1] : srcSize[0]);

	int A_rows = mxGetM(src);
	int A_cols = mxGetN(src);

	//-----------------------------------------------------------------------//
	// configure output
	int B_dim[2];
	B_dim[0] = A_rows;
	B_dim[1] = A_cols;
	plhs[0] = mxCreateNumericArray(2, B_dim, mxSINGLE_CLASS, mxREAL);
	float *B = (float*)mxGetData(plhs[0]);

	//-----------------------------------------------------------------------//
	// create gpu buffers
	int len = A_rows * A_cols;
	float *d_A;
	cudaMalloc((void**)&d_A, len * sizeof(float));
	float *d_B;
	cudaMalloc((void**)&d_B, len * sizeof(float));

	// copy A to gpu
	cudaMemcpy(d_A, A, len * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blk_dims = dim3(256,1,1);
	dim3 grd_dims = dim3(BLK(len, 256),1,1);

	// execute kernel
	kernel_vectoradd <<< grd_dims, blk_dims >>>(d_A, len, d_B);

	// copy data back
	cudaMemcpy(B, d_B, len * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
}

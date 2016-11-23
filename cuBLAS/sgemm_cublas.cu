#include "mex.h"

#include <cuda_runtime.h>                                                       
#include <helper_cuda.h>                                                        
#include <helper_functions.h>

#include <cublas_v2.h>

#define FLT_SIZE sizeof(float)
#define INT_SIZE sizeof(int)

inline int BLK(int number, int blksize)                                         
{
	return (number + blksize - 1) / blksize;
}

/*
C = sgemm_cublas(transa, 
				transb, 
				single(alpha), 
				single(beta), 
				single(A), 
				single(B));

transa,transb = 0 (no transpose) or 1 (transpose) of A, B

Input arrays are in single precision.
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	//-----------------------------------------------------------------------//
	if (nrhs != 7) {
		mexErrMsgTxt("sgemm_cublas requires 7 input arguments");
		exit(0);
	} else if (nlhs != 1) {
		mexErrMsgTxt("sgemm_cublas requires 1 output argument");
		exit(0);
	}


	if(!mxIsSingle(prhs[4]) || !mxIsSingle(prhs[5]) || !mxIsSingle(prhs[6])) {
		mexErrMsgTxt("Input arrays must be single precision.");
		exit(0);
	}

	//-----------------------------------------------------------------------//
	// init cublas
	cublasStatus_t status;

	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		exit(1);
	}


	//-----------------------------------------------------------------------//
	int ta, tb;
	ta = (int) mxGetScalar(prhs[0]);
	tb = (int) mxGetScalar(prhs[1]);

	float alpha, beta;
	alpha = (float) mxGetScalar(prhs[2]);
	beta  = (float) mxGetScalar(prhs[3]);

	int M,K,L,N;

	M = mxGetM(prhs[4]); // rows of A
	K = mxGetN(prhs[4]); // cols of A

	L = mxGetM(prhs[5]); // rows of B
	N = mxGetN(prhs[5]); // cols of B


	cublasOperation_t transa, transb;
	int MM, KK, NN;

	// configure A
	if(ta == 0) {
		transa = CUBLAS_OP_N;	
		MM = M;
		KK = K;
	} else {
		transa = CUBLAS_OP_T;	
		MM = K;
		KK = M;
	}

	// configure B
	if(tb == 0) {
		transb = CUBLAS_OP_N;	
		NN = N;	// NN is the column of B
	} else {
		transa = CUBLAS_OP_T;	
		NN = L;
	}

	// set up input array
	float *A = (float*)mxGetData(prhs[4]);
	float *B = (float*)mxGetData(prhs[5]);
	float *C = (float*)mxGetData(prhs[6]);

	// create output array
	int C_dim[2];
	C_dim[0] = MM;
	C_dim[1] = NN;
	plhs[0] = mxCreateNumericArray(2, C_dim, mxSINGLE_CLASS, mxREAL);
	float *C_out = (float*)mxGetData(plhs[0]);

	//-----------------------------------------------------------------------//
	// allocate gpu buffers

	// configure d_A
	// round to 32
	int Mc = (BLK(M, 32) << 5); 
	int Kc = (BLK(K, 32) << 5); 
	float *d_A;
	checkCudaErrors(cudaMalloc((void **)&d_A,  Mc * Kc * FLT_SIZE));
	// zero out the array
	cudaMemset(d_A, 0, Mc * Kc * FLT_SIZE);

	// configure d_B
	int Lc = (BLK(L, 32) << 5); 	// rows
	int Nc = (BLK(N, 32) << 5); 
	float *d_B;
	checkCudaErrors(cudaMalloc((void **)&d_B,  Lc * Nc * FLT_SIZE));
	cudaMemset(d_B, 0, Lc * Nc * FLT_SIZE);

	// allocate d_C
	int MMc = (BLK(MM, 32) << 5);
	int NNc = (BLK(NN, 32) << 5); 
	int KKc = (BLK(KK, 32) << 5); 
	float *d_C;
	checkCudaErrors(cudaMalloc((void **)&d_C,  MMc * NNc * FLT_SIZE));


	//-----------------------------------------------------------------------//
	// copy data to gpu

	// d_A
	status = cublasSetMatrix(M, K, FLT_SIZE, A, M, d_A, Mc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write A)\n");
		exit(1);
	}

	// d_B
	status = cublasSetMatrix(L, N, FLT_SIZE, B, L, d_B, Lc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write B)\n");
		exit(1);
	}

	// d_C
	if(beta != 0.f) {
		cudaMemset(d_C, 0, MMc * NNc * FLT_SIZE);
		status = cublasSetMatrix(MM, NN, FLT_SIZE, C, MM, d_C, MMc);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! device access error (write C)\n");
			exit(1);
		}
	}

	//-----------------------------------------------------------------------//
	// call cublas sgemm
	//-----------------------------------------------------------------------//
	status = cublasSgemm(handle,
			transa,
			transb,
			MMc, NNc, KKc,
			&alpha,
			d_A, Mc,
			d_B, Lc,
			&beta,
			d_C, MMc);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! cublasSgemm execution error.\n");
		exit(1);
	}

	//-----------------------------------------------------------------------//
	// copy d_C to C_out
	status = cublasGetMatrix(M, N, FLT_SIZE, d_C, MMc, C_out, MM);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! cublasGetMatrix error (C_out).\n");
		exit(1);
	}


	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	// cublas shutdown
	cublasDestroy(handle);
}

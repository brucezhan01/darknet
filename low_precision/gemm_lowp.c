#include "gemm_lowp.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#if 0
void gemm_bin(int M, int N, int K, lowp_t ALPHA, 
        char  *A, int lda, 
        lowp_t *B, int ldb,
        lowp_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

lowp_t *random_matrix(int rows, int cols)
{
    int i;
    lowp_t *m = (lowp_t*)calloc(rows*cols, sizeof(lowp_t));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (lowp_t)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    lowp_t *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    lowp_t *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    lowp_t *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,(lowp_t)1,a,lda,b,ldb,(lowp_t)1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (lowp_t)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}
#endif

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int m, n, k;

    /*
    lowp_t *lowpA = new lowp_t[M*K*sizeof(lowp_t)];
    lowp_t *lowpB = new lowp_t[K*N*sizeof(lowp_t)];
    lowp_t *lowpC = new lowp_t[M*N*sizeof(lowp_t)];
    */
    lowp_t *lowpA = (lowp_t*)malloc(M*K*sizeof(lowp_t));
    lowp_t *lowpB = (lowp_t*)malloc(K*N*sizeof(lowp_t));
    acc_lowp_t *lowpC = (acc_lowp_t*)malloc(M*N*sizeof(acc_lowp_t));
    
    //memset(lowpC, 0, M*N*sizeof(lowp_t));
    for (m = 0; m < M; ++m) {
        for (n = 0; n < N; ++n) {
            lowpC[m*ldc+n] = (acc_lowp_t)((C[m*ldc+n]));
            //lowpC[m*ldc+n] = half_float::half_cast<half, std::round_to_nearest>(C[m*ldc+n]);
        }
    }
    
    for(m = 0; m < M; ++m){
        for(k = 0; k < K; ++k){
            lowpA[m*lda+k] = (lowp_t)(A[m*lda+k]);
            //lowpA[m*lda+k] = half_float::half_cast<half, std::round_to_nearest>(A[m*lda+k]);
        }
    }
    for(k = 0; k < K; ++k){
        for(n = 0; n < N; ++n){
            lowpB[k*ldb+n] = (lowp_t)(B[k*ldb+n]);
            //lowpB[k*ldb+n] = half_float::half_cast<half, std::round_to_nearest>(B[k*ldb+n]);
        }
    }
    
    gemm_cpu(TA, TB, M, N, K, (lowp_t)ALPHA, lowpA, lda, lowpB, ldb, (lowp_t)BETA, lowpC, ldc);


    for (m = 0; m < M; ++m) {
        for (n = 0; n < N; ++n) {
            C[m*ldc+n] = (float)((lowp_t)(lowpC[m*ldc+n]));
            //C[m*ldc+n] = lowpC[m*ldc+n];
        }
    }

    
    /*
    delete []lowpA;
    delete []lowpB;
    delete []lowpC;
    */
    free(lowpA);
    free(lowpB);
    free(lowpC);
}

void gemm_nn(int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        acc_lowp_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register lowp_t A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        acc_lowp_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register lowp_t sum = (lowp_t)0.0f;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        acc_lowp_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register lowp_t A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        acc_lowp_t *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register lowp_t sum = (lowp_t)0.0f;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        lowp_t BETA,
        acc_lowp_t *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_ongpu(int TA, int TB, int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A_gpu, int lda, 
        lowp_t *B_gpu, int ldb,
        lowp_t BETA,
        lowp_t *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_cublas_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        lowp_t BETA,
        lowp_t *C, int ldc)
{
    lowp_t *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    lowp_t *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    lowp_t *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    lowp_t *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    lowp_t *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    lowp_t *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (lowp_t)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    lowp_t *a = random_matrix(m,k);
    lowp_t *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    lowp_t *c = random_matrix(m,n);

    lowp_t *a_cl = cuda_make_array(a, m*k);
    lowp_t *b_cl = cuda_make_array(b, k*n);
    lowp_t *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    lowp_t *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    lowp_t *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    lowp_t *c = random_matrix(m,n);
    lowp_t *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(lowp_t));
    memset(c_gpu, 0, m*n*sizeof(lowp_t));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,192,729,1600); 
       time_ongpu(0,0,384,196,1728); 
       time_ongpu(0,0,256,196,3456); 
       time_ongpu(0,0,256,196,2304); 
       time_ongpu(0,0,128,4096,12544); 
       time_ongpu(0,0,128,4096,4096); 
     */
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,576,12544); 
    time_ongpu(0,0,256,2304,784); 
    time_ongpu(1,1,2304,256,784); 
    time_ongpu(0,0,512,4608,196); 
    time_ongpu(1,1,4608,512,196); 

    return 0;
}
#endif


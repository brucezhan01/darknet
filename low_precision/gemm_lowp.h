#ifndef GEMM_LOWP_H
#define GEMM_LOWP_H

#include "half.hpp"

using half_float::half;
using half_float::half_cast;

typedef half lowp_t;
typedef float acc_lowp_t;

void gemm_cpu(int TA, int TB, int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        lowp_t BETA,
        acc_lowp_t *C, int ldc);
        
#endif /* GEMM_LOWP_H */

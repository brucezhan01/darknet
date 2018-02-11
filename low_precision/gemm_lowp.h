#ifndef GEMM_LOWP_H
#define GEMM_LOWP_H

#include "half.hpp"
#include "cpfp.hpp"

using half_float::half;
using half_float::half_cast;

#if 1
typedef cpfp lowp_t;
typedef half acc_lowp_t;

#define LOWP_CAST(v)         ((cpfp)(v))
#define LOWP_PTR_CAST(v)     ((cpfp*)(v))
#define ACC_LOWP_CAST(v)      ((half)(v)) //((cpfp)(v))
#define ACC_LOWP_PTR_CAST(v)  ((half*)(v)) //((cpfp*)(v))
#else
typedef half lowp_t;
typedef half acc_lowp_t;

#define LOWP_CAST(v)         (half_cast<half, std::round_to_nearest>(v))
#define LOWP_PTR_CAST(v)     ((half*)(v))
#define ACC_LOWP_CAST(v)      ((half)(v))
#define ACC_LOWP_PTR_CAST(v)  ((half*)(v))    
#endif

void gemm_cpu(int TA, int TB, int M, int N, int K, lowp_t ALPHA, 
        lowp_t *A, int lda, 
        lowp_t *B, int ldb,
        lowp_t BETA,
        acc_lowp_t *C, int ldc);
        
#endif /* GEMM_LOWP_H */

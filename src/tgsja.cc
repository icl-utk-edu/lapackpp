#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tgsja(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n, int64_t k, int64_t l,
    float* A, int64_t lda,
    float* B, int64_t ldb, float tola, float tolb,
    float* alpha,
    float* beta,
    float* U, int64_t ldu,
    float* V, int64_t ldv,
    float* Q, int64_t ldq,
    int64_t* ncycle )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job_compu2char( jobu );
    char jobv_ = job_comp2char( jobv );
    char jobq_ = job_compq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int l_ = (blas_int) l;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ncycle_ = (blas_int) *ncycle;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (2*n) );

    LAPACK_stgsja(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, &k_, &l_,
        A, &lda_,
        B, &ldb_, &tola, &tolb,
        alpha,
        beta,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        &work[0], &ncycle_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ncycle = ncycle_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsja(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n, int64_t k, int64_t l,
    double* A, int64_t lda,
    double* B, int64_t ldb, double tola, double tolb,
    double* alpha,
    double* beta,
    double* U, int64_t ldu,
    double* V, int64_t ldv,
    double* Q, int64_t ldq,
    int64_t* ncycle )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job_compu2char( jobu );
    char jobv_ = job_comp2char( jobv );
    char jobq_ = job_compq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int l_ = (blas_int) l;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ncycle_ = (blas_int) *ncycle;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (2*n) );

    LAPACK_dtgsja(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, &k_, &l_,
        A, &lda_,
        B, &ldb_, &tola, &tolb,
        alpha,
        beta,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        &work[0], &ncycle_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ncycle = ncycle_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsja(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n, int64_t k, int64_t l,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb, float tola, float tolb,
    float* alpha,
    float* beta,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* V, int64_t ldv,
    std::complex<float>* Q, int64_t ldq,
    int64_t* ncycle )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job_compu2char( jobu );
    char jobv_ = job_comp2char( jobv );
    char jobq_ = job_compq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int l_ = (blas_int) l;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ncycle_ = (blas_int) *ncycle;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );

    LAPACK_ctgsja(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, &k_, &l_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &tola, &tolb,
        alpha,
        beta,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) &work[0], &ncycle_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ncycle = ncycle_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsja(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n, int64_t k, int64_t l,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb, double tola, double tolb,
    double* alpha,
    double* beta,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* V, int64_t ldv,
    std::complex<double>* Q, int64_t ldq,
    int64_t* ncycle )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job_compu2char( jobu );
    char jobv_ = job_comp2char( jobv );
    char jobq_ = job_compq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int l_ = (blas_int) l;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ncycle_ = (blas_int) *ncycle;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );

    LAPACK_ztgsja(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, &k_, &l_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &tola, &tolb,
        alpha,
        beta,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) &work[0], &ncycle_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ncycle = ncycle_;
    return info_;
}

}  // namespace lapack

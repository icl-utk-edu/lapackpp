#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geqp3(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    int64_t* jpvt,
    float* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqp3(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    int64_t* jpvt,
    double* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqp3(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* jpvt,
    std::complex<float>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (2*n) );

    LAPACK_cgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqp3(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* jpvt,
    std::complex<double>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (2*n) );

    LAPACK_zgeqp3( &m_, &n_, A, &lda_, jpvt_ptr, TAU, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    return info_;
}

}  // namespace lapack

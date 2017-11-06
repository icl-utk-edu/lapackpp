#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geqr(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(tsize) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int tsize_ = (blas_int) tsize;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgeqr( &m_, &n_, A, &lda_, T, &tsize_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgeqr( &m_, &n_, A, &lda_, T, &tsize_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqr(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(tsize) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int tsize_ = (blas_int) tsize;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgeqr( &m_, &n_, A, &lda_, T, &tsize_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgeqr( &m_, &n_, A, &lda_, T, &tsize_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqr(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(tsize) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int tsize_ = (blas_int) tsize;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cgeqr( &m_, &n_, A, &lda_, T, &tsize_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgeqr( &m_, &n_, A, &lda_, T, &tsize_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqr(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(tsize) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int tsize_ = (blas_int) tsize;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zgeqr( &m_, &n_, A, &lda_, T, &tsize_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgeqr( &m_, &n_, A, &lda_, T, &tsize_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

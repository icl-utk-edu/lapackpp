#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    float* D,
    float* E,
    float* VT, int64_t ldvt,
    float* U, int64_t ldu,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nru) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ncvt_ = (blas_int) ncvt;
    blas_int nru_ = (blas_int) nru;
    blas_int ncc_ = (blas_int) ncc;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (4*n) );

    LAPACK_sbdsqr( &uplo_, &n_, &ncvt_, &nru_, &ncc_, D, E, VT, &ldvt_, U, &ldu_, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    double* D,
    double* E,
    double* VT, int64_t ldvt,
    double* U, int64_t ldu,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nru) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ncvt_ = (blas_int) ncvt;
    blas_int nru_ = (blas_int) nru;
    blas_int ncc_ = (blas_int) ncc;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (4*n) );

    LAPACK_dbdsqr( &uplo_, &n_, &ncvt_, &nru_, &ncc_, D, E, VT, &ldvt_, U, &ldu_, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    float* D,
    float* E,
    std::complex<float>* VT, int64_t ldvt,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nru) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ncvt_ = (blas_int) ncvt;
    blas_int nru_ = (blas_int) nru;
    blas_int ncc_ = (blas_int) ncc;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > rwork( (4*n) );

    LAPACK_cbdsqr( &uplo_, &n_, &ncvt_, &nru_, &ncc_, D, E, VT, &ldvt_, U, &ldu_, C, &ldc_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    double* D,
    double* E,
    std::complex<double>* VT, int64_t ldvt,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nru) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ncvt_ = (blas_int) ncvt;
    blas_int nru_ = (blas_int) nru;
    blas_int ncc_ = (blas_int) ncc;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > rwork( (4*n) );

    LAPACK_zbdsqr( &uplo_, &n_, &ncvt_, &nru_, &ncc_, D, E, VT, &ldvt_, U, &ldu_, C, &ldc_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

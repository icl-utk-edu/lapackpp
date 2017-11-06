#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    float const* DL,
    float const* D,
    float const* DU,
    float const* DLF,
    float const* DF,
    float const* DUF,
    float const* DU2,
    int64_t const* ipiv,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    double const* DL,
    double const* D,
    double const* DU,
    double const* DLF,
    double const* DF,
    double const* DUF,
    double const* DU2,
    int64_t const* ipiv,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU,
    std::complex<float> const* DLF,
    std::complex<float> const* DF,
    std::complex<float> const* DUF,
    std::complex<float> const* DU2,
    int64_t const* ipiv,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_cgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU,
    std::complex<double> const* DLF,
    std::complex<double> const* DF,
    std::complex<double> const* DUF,
    std::complex<double> const* DU2,
    int64_t const* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_zgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

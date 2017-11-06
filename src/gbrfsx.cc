#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    float const* AB, int64_t ldab,
    float const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    float* R,
    float* C,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* ERR_BNDS_NORM,
    float* ERR_BNDS_COMP, int64_t nparams,
    float* PARAMS )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldafb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldafb_ = (blas_int) ldafb;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (4*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sgbrfsx( &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_, AB, &ldab_, AFB, &ldafb_, ipiv_ptr, R, C, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, ERR_BNDS_NORM, ERR_BNDS_COMP, &nparams_, PARAMS, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    double const* AB, int64_t ldab,
    double const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    double* R,
    double* C,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* ERR_BNDS_NORM,
    double* ERR_BNDS_COMP, int64_t nparams,
    double* PARAMS )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldafb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldafb_ = (blas_int) ldafb;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (4*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dgbrfsx( &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_, AB, &ldab_, AFB, &ldafb_, ipiv_ptr, R, C, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, ERR_BNDS_NORM, ERR_BNDS_COMP, &nparams_, PARAMS, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<float> const* AB, int64_t ldab,
    std::complex<float> const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    float* R,
    float* C,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* ERR_BNDS_NORM,
    float* ERR_BNDS_COMP, int64_t nparams,
    float* PARAMS )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldafb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldafb_ = (blas_int) ldafb;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (2*n) );

    LAPACK_cgbrfsx( &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_, AB, &ldab_, AFB, &ldafb_, ipiv_ptr, R, C, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, ERR_BNDS_NORM, ERR_BNDS_COMP, &nparams_, PARAMS, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<double> const* AB, int64_t ldab,
    std::complex<double> const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    double* R,
    double* C,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* ERR_BNDS_NORM,
    double* ERR_BNDS_COMP, int64_t nparams,
    double* PARAMS )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldafb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldafb_ = (blas_int) ldafb;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (2*n) );

    LAPACK_zgbrfsx( &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_, AB, &ldab_, AFB, &ldafb_, ipiv_ptr, R, C, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, ERR_BNDS_NORM, ERR_BNDS_COMP, &nparams_, PARAMS, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

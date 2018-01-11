#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::CompQ compq, int64_t n,
    float* T, int64_t ldt,
    float* Q, int64_t ldq,
    int64_t* ifst,
    int64_t* ilst )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ifst_ = (blas_int) *ifst;
    blas_int ilst_ = (blas_int) *ilst;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n) );

    LAPACK_strexc( &compq_, &n_, T, &ldt_, Q, &ldq_, &ifst_, &ilst_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ifst = ifst_;
    *ilst = ilst_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::CompQ compq, int64_t n,
    double* T, int64_t ldt,
    double* Q, int64_t ldq,
    int64_t* ifst,
    int64_t* ilst )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ifst_ = (blas_int) *ifst;
    blas_int ilst_ = (blas_int) *ilst;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n) );

    LAPACK_dtrexc( &compq_, &n_, T, &ldt_, Q, &ldq_, &ifst_, &ilst_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ifst = ifst_;
    *ilst = ilst_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::CompQ compq, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* Q, int64_t ldq, int64_t ifst, int64_t ilst )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ifst) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilst) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ifst_ = (blas_int) ifst;
    blas_int ilst_ = (blas_int) ilst;
    blas_int info_ = 0;

    LAPACK_ctrexc( &compq_, &n_, T, &ldt_, Q, &ldq_, &ifst_, &ilst_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::CompQ compq, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* Q, int64_t ldq, int64_t ifst, int64_t ilst )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ifst) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilst) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ifst_ = (blas_int) ifst;
    blas_int ilst_ = (blas_int) ilst;
    blas_int info_ = 0;

    LAPACK_ztrexc( &compq_, &n_, T, &ldt_, Q, &ldq_, &ifst_, &ilst_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

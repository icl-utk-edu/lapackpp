#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t spgvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    float* AP,
    float* BP, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* m,
    float* W,
    float* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int m_ = (blas_int) *m;
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (n) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (8*n) );
    std::vector< blas_int > iwork( (5*n) );

    LAPACK_sspgvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        AP,
        BP, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t spgvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    double* AP,
    double* BP, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* m,
    double* W,
    double* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int m_ = (blas_int) *m;
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (n) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (8*n) );
    std::vector< blas_int > iwork( (5*n) );

    LAPACK_dspgvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        AP,
        BP, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

}  // namespace lapack

#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup heev
int64_t syevr_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* nfound,
    float* W,
    float* Z, int64_t ldz,
    int64_t* isuppz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_;  // output
    lapack_int ldz_ = (lapack_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > isuppz_( (2*max( 1, n )) );  // was max(1,m), n >= m
        lapack_int* isuppz_ptr = &isuppz_[0];
    #else
        lapack_int* isuppz_ptr = isuppz;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_ssyevr_2stage(
        &jobz_, &range_, &uplo_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        isuppz_ptr,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< lapack_int > iwork( liwork_ );

    LAPACK_ssyevr_2stage(
        &jobz_, &range_, &uplo_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        isuppz_ptr,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #if 1
        std::copy( isuppz_.begin(), isuppz_.end(), isuppz );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @see lapack::heevr_2stage
/// @ingroup heev
int64_t syevr_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* nfound,
    double* W,
    double* Z, int64_t ldz,
    int64_t* isuppz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_;  // output
    lapack_int ldz_ = (lapack_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > isuppz_( (2*max( 1, n )) );  // was max(1,m), n >= m
        lapack_int* isuppz_ptr = &isuppz_[0];
    #else
        lapack_int* isuppz_ptr = isuppz;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dsyevr_2stage(
        &jobz_, &range_, &uplo_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        isuppz_ptr,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< lapack_int > iwork( liwork_ );

    LAPACK_dsyevr_2stage(
        &jobz_, &range_, &uplo_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        isuppz_ptr,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #if 1
        std::copy( isuppz_.begin(), isuppz_.end(), isuppz );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7

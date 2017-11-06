#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t heev_2stage(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
    float* W )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cheev_2stage( &jobz_, &uplo_, &n_, A, &lda_, W, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (max( (int64_t) 1, 3*n-2)) );

    LAPACK_cheev_2stage( &jobz_, &uplo_, &n_, A, &lda_, W, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t heev_2stage(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
    double* W )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zheev_2stage( &jobz_, &uplo_, &n_, A, &lda_, W, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (max( (int64_t) 1, 3*n-2)) );

    LAPACK_zheev_2stage( &jobz_, &uplo_, &n_, A, &lda_, W, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

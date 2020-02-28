#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t stevd(
    lapack::Job jobz, int64_t n,
    float* D,
    float* E,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_sstevd(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
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

    LAPACK_sstevd(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stevd(
    lapack::Job jobz, int64_t n,
    double* D,
    double* E,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dstevd(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
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

    LAPACK_dstevd(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

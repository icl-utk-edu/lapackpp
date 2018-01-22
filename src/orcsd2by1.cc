#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION >= 30500  // >= 3.5

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t orcsd2by1(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, int64_t m, int64_t p, int64_t q,
    float* X11, int64_t ldx11,
    float* X21, int64_t ldx21,
    float* THETA,
    float* U1, int64_t ldu1,
    float* U2, int64_t ldu2,
    float* V1T, int64_t ldv1t )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx11) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx21) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldx11_ = (blas_int) ldx11;
    blas_int ldx21_ = (blas_int) ldx21;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_sorcsd2by1( &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_, X11, &ldx11_, X21, &ldx21_, THETA, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( (m - min( p, min( m-p, min( q, m-q )))) );

    LAPACK_sorcsd2by1( &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_, X11, &ldx11_, X21, &ldx21_, THETA, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t orcsd2by1(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, int64_t m, int64_t p, int64_t q,
    double* X11, int64_t ldx11,
    double* X21, int64_t ldx21,
    double* THETA,
    double* U1, int64_t ldu1,
    double* U2, int64_t ldu2,
    double* V1T, int64_t ldv1t )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx11) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx21) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldx11_ = (blas_int) ldx11;
    blas_int ldx21_ = (blas_int) ldx21;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dorcsd2by1( &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_, X11, &ldx11_, X21, &ldx21_, THETA, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( (m - min( p, min( m-p, min( q, m-q )))) );

    LAPACK_dorcsd2by1( &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_, X11, &ldx11_, X21, &ldx21_, THETA, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.5.0

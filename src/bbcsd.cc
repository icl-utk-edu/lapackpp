#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 3  // >= 3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t bbcsd(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, lapack::JobCS jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    float* THETA,
    float* PHI,
    float* U1, int64_t ldu1,
    float* U2, int64_t ldu2,
    float* V1T, int64_t ldv1t,
    float* V2T, int64_t ldv2t,
    float* B11D,
    float* B11E,
    float* B12D,
    float* B12E,
    float* B21D,
    float* B21E,
    float* B22D,
    float* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(q) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv2t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    char jobv2t_ = jobcs2char( jobv2t );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int ldv2t_ = (blas_int) ldv2t;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bbcsd(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, lapack::JobCS jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    double* THETA,
    double* PHI,
    double* U1, int64_t ldu1,
    double* U2, int64_t ldu2,
    double* V1T, int64_t ldv1t,
    double* V2T, int64_t ldv2t,
    double* B11D,
    double* B11E,
    double* B12D,
    double* B12E,
    double* B21D,
    double* B21E,
    double* B22D,
    double* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(q) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv2t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    char jobv2t_ = jobcs2char( jobv2t );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int ldv2t_ = (blas_int) ldv2t;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bbcsd(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, lapack::JobCS jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    float* THETA,
    float* PHI,
    std::complex<float>* U1, int64_t ldu1,
    std::complex<float>* U2, int64_t ldu2,
    std::complex<float>* V1T, int64_t ldv1t,
    std::complex<float>* V2T, int64_t ldv2t,
    float* B11D,
    float* B11E,
    float* B12D,
    float* B12E,
    float* B21D,
    float* B21E,
    float* B22D,
    float* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(q) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv2t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    char jobv2t_ = jobcs2char( jobv2t );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int ldv2t_ = (blas_int) ldv2t;
    blas_int info_ = 0;

    // query for workspace size
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, qry_rwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    std::vector< float > rwork( lrwork_ );

    LAPACK_cbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, &rwork[0], &lrwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bbcsd(
    lapack::JobCS jobu1, lapack::JobCS jobu2, lapack::JobCS jobv1t, lapack::JobCS jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    double* THETA,
    double* PHI,
    std::complex<double>* U1, int64_t ldu1,
    std::complex<double>* U2, int64_t ldu2,
    std::complex<double>* V1T, int64_t ldv1t,
    std::complex<double>* V2T, int64_t ldv2t,
    double* B11D,
    double* B11E,
    double* B12D,
    double* B12E,
    double* B21D,
    double* B21E,
    double* B22D,
    double* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(q) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv1t) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv2t) > std::numeric_limits<blas_int>::max() );
    }
    char jobu1_ = jobcs2char( jobu1 );
    char jobu2_ = jobcs2char( jobu2 );
    char jobv1t_ = jobcs2char( jobv1t );
    char jobv2t_ = jobcs2char( jobv2t );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int q_ = (blas_int) q;
    blas_int ldu1_ = (blas_int) ldu1;
    blas_int ldu2_ = (blas_int) ldu2;
    blas_int ldv1t_ = (blas_int) ldv1t;
    blas_int ldv2t_ = (blas_int) ldv2t;
    blas_int info_ = 0;

    // query for workspace size
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, qry_rwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    std::vector< double > rwork( lrwork_ );

    LAPACK_zbbcsd( &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_, THETA, PHI, U1, &ldu1_, U2, &ldu2_, V1T, &ldv1t_, V2T, &ldv2t_, B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E, &rwork[0], &lrwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.3.0

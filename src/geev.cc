#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    float* A, int64_t lda,
    std::complex<float>* W,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // split-complex representation
    std::vector< float > WR( max( 1, n ) );
    std::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    double* A, int64_t lda,
    std::complex<double>* W,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // split-complex representation
    std::vector< double > WR( max( 1, n ) );
    std::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* W,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (2*n) );

    LAPACK_cgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes for an n-by-n nonsymmetric matrix A, the
/// eigenvalues and, optionally, the left and/or right eigenvectors.
///
/// The right eigenvector v_j of A satisfies
///     \f$ A v_j = \lambda_j v_j \f$
/// where \f$ lambda_j \f$ is its eigenvalue.
/// The left eigenvector u_j of A satisfies
///     \f$ u_j^H A = \lambda_j u_j^H \f$
/// where \f$ u_j^H \f$ denotes the conjugate transpose of \f$ u_j \f$.
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to 1 and largest component real.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobvl
///     - lapack::Job::NoVec: left eigenvectors of A are not computed;
///     - lapack::Job::Vec:   left eigenvectors of are computed.
///
/// @param[in] jobvr
///     - lapack::Job::NoVec: right eigenvectors of A are not computed;
///     - lapack::Job::Vec:   right eigenvectors of A are computed.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the n-by-n matrix A.
///     On exit, A has been overwritten.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] W
///     The vector W of length n.
///     W contains the computed eigenvalues.
///     [Note: In LAPACK++, W is always complex, whereas with real matrices,
///     LAPACK uses a split-complex representation (WR,WI) for W.]
///
/// @param[out] VL
///     The n-by-n matrix VL, stored in an ldvl-by-n array.
///     \n
///     If jobvl = Vec, the left eigenvectors \f$ u_j \f$ are stored one
///     after another in the columns of VL, in the same order
///     as their eigenvalues.
///     \n
///     If jobvl = NoVec, VL is not referenced.
///     \n
///     For std::complex versions:
///     \f$ u_j \f$ = VL(:,j), the j-th column of VL.
///     \n
///     For real (float, double) versions:
///     If the j-th eigenvalue is real, then
///     \f$ u_j \f$ = VL(:,j),
///     the j-th column of VL.
///     If the j-th and (j+1)-st eigenvalues form a complex
///     conjugate pair, then
///     \f$ u_j     \f$ = VL(:,j) + i*VL(:,j+1) and
///     \f$ u_{j+1} \f$ = VL(:,j) - i*VL(:,j+1).
///
/// @param[in] ldvl
///     The leading dimension of the array VL. ldvl >= 1;
///     if jobvl = Vec, ldvl >= n.
///
/// @param[out] VR
///     The n-by-n matrix VR, stored in an ldvr-by-n array.
///     \n
///     If jobvr = Vec, the right eigenvectors \f$ v_j \f$ are stored one
///     after another in the columns of VR, in the same order
///     as their eigenvalues.
///     \n
///     If jobvr = NoVec, VR is not referenced.
///     \n
///     For std::complex versions:
///     \f$ v_j \f$ = VR(:,j), the j-th column of VR.
///     \n
///     For real (float, double) versions:
///     If the j-th eigenvalue is real, then
///     \f$ v_j \f$ = VR(:,j),
///     the j-th column of VR.
///     If the j-th and (j+1)-st eigenvalues form a complex
///     conjugate pair, then
///     \f$ v_j     \f$ = VR(:,j) + i*VR(:,j+1) and
///     \f$ v_{j+1} \f$ = VR(:,j) - i*VR(:,j+1).
///
/// @param[in] ldvr
///     The leading dimension of the array VR. ldvr >= 1;
///     if jobvr = Vec, ldvr >= n.
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, the QR algorithm failed to compute all the
///              eigenvalues, and no eigenvectors have been computed;
///              elements i+1:n of W contain eigenvalues which have
///              converged.
///
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* W,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (2*n) );

    LAPACK_zgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 6 && LAPACK_VERSION_MICRO >=1  // >= 3.6.1

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t trevc3(
    lapack::Sides side, lapack::HowMany howmany,
    bool* select, int64_t n,
    float const* T, int64_t ldt,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(mm) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = sides2char( side );
    char howmany_ = howmany2char( howmany );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int* select_ptr = &select_[0];
    #else
        blas_int* select_ptr = select;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int mm_ = (blas_int) mm;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_strevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_strevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( select_.begin(), select_.end(), select );
    #endif
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t trevc3(
    lapack::Sides side, lapack::HowMany howmany,
    bool* select, int64_t n,
    double const* T, int64_t ldt,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(mm) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = sides2char( side );
    char howmany_ = howmany2char( howmany );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int* select_ptr = &select_[0];
    #else
        blas_int* select_ptr = select;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int mm_ = (blas_int) mm;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dtrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dtrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( select_.begin(), select_.end(), select );
    #endif
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t trevc3(
    lapack::Sides side, lapack::HowMany howmany,
    bool const* select, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(mm) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = sides2char( side );
    char howmany_ = howmany2char( howmany );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int mm_ = (blas_int) mm;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_ctrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, qry_work, &ineg_one, qry_rwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( lrwork_ );

    LAPACK_ctrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &lwork_, &rwork[0], &lrwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes some or all of the right and/or left eigenvectors of
/// a complex upper triangular matrix T.
/// Matrices of this type are produced by the Schur factorization of
/// a complex general matrix: \f$ A = Q T Q^H \f$, as computed by `lapack::hseqr`.
///
/// The right eigenvector x and the left eigenvector y of T corresponding
/// to an eigenvalue \f$ \lambda \f$ are defined by:
///
///     \f[ T x = \lambda x \f]
///     \f[ y^H T = \lambda y^H \f]
///
/// where \f$ y^H \f$ denotes the conjugate transpose of the vector y.
/// The eigenvalues are not input to this routine, but are read directly
/// from the diagonal of T.
///
/// This routine returns the matrices X and/or Y of right and left
/// eigenvectors of T, or the products \f$ Q X \f$ and/or \f$ Q Y \f$, where Q is an
/// input matrix. If Q is the unitary factor that reduces a matrix A to
/// Schur form T, then \f$ Q X \f$ and \f$ Q Y \f$ are the matrices of right and left
/// eigenvectors of A.
///
/// This uses a Level 3 BLAS version of the back transformation.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     - lapack::Sides::Right: compute right eigenvectors only;
///     - lapack::Sides::Left:  compute left eigenvectors only;
///     - lapack::Sides::Both:  compute both right and left eigenvectors.
///
/// @param[in] howmany
///     - lapack::HowMany::All:
///         compute all right and/or left eigenvectors;
///     - lapack::HowMany::Backtransform:
///         compute all right and/or left eigenvectors,
///         backtransformed using the matrices supplied in
///         VR and/or VL;
///     - lapack::HowMany::Select:
///         compute selected right and/or left eigenvectors,
///         as indicated by the logical array select.
///
/// @param[in] select
///     The vector select of length n.
///     If howmany = Select, select specifies the eigenvectors to be
///     computed.
///     The eigenvector corresponding to the j-th eigenvalue is
///     computed if select(j) = true.
///     Not referenced if howmany = All or Backtransform.
///
/// @param[in] n
///     The order of the matrix T. n >= 0.
///
/// @param[in,out] T
///     The n-by-n matrix T, stored in an ldt-by-n array.
///     The upper triangular matrix T. T is modified, but restored
///     on exit.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= max(1,n).
///
/// @param[in,out] VL
///     The n-by-mm matrix VL, stored in an ldvl-by-mm array.
///     On entry, if side = Left or Both and howmany = Backtransform, VL must
///     contain an n-by-n matrix Q (usually the unitary matrix Q of
///     Schur vectors returned by `lapack::hseqr`).
///     On exit, if side = Left or Both, VL contains:
///     - if howmany = All, the matrix Y of left eigenvectors of T;
///     - if howmany = Backtransform, the matrix \f$ Q Y \f$;
///     - if howmany = Select, the left eigenvectors of T specified by
///         select, stored consecutively in the columns
///         of VL, in the same order as their
///         eigenvalues.
///     \n
///     Not referenced if side = Right.
///
/// @param[in] ldvl
///     The leading dimension of the array VL.
///     ldvl >= 1, and if side = Left or Both, ldvl >= n.
///
/// @param[in,out] VR
///     The n-by-mm matrix VR, stored in an ldvr-by-mm array.
///     On entry, if side = Right or Both and howmany = Backtransform, VR must
///     contain an n-by-n matrix Q (usually the unitary matrix Q of
///     Schur vectors returned by `lapack::hseqr`).
///     On exit, if side = Right or Both, VR contains:
///     - if howmany = All, the matrix X of right eigenvectors of T;
///     - if howmany = Backtransform, the matrix \f$ Q X \f$;
///     - if howmany = Select, the right eigenvectors of T specified by
///         select, stored consecutively in the columns
///         of VR, in the same order as their
///         eigenvalues.
///     \n
///     Not referenced if side = Left.
///
/// @param[in] ldvr
///     The leading dimension of the array VR.
///     ldvr >= 1, and if side = Right or Both, ldvr >= n.
///
/// @param[in] mm
///     The number of columns in the arrays VL and/or VR. mm >= m.
///
/// @param[out] m
///     The number of columns in the arrays VL and/or VR actually
///     used to store the eigenvectors.
///     If howmany = All or Backtransform, m is set to n.
///     Each selected eigenvector occupies one column.
///
/// @retval = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The algorithm used in this program is basically backward (forward)
/// substitution, with scaling to make the the code robust against
/// possible overflow.
///
/// Each eigenvector is normalized so that the element of largest
/// magnitude has magnitude 1; here the magnitude of a complex number
/// (x,y) is taken to be |x| + |y|.
///
/// @ingroup geev_computational
int64_t trevc3(
    lapack::Sides side, lapack::HowMany howmany,
    bool const* select, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(mm) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = sides2char( side );
    char howmany_ = howmany2char( howmany );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int mm_ = (blas_int) mm;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_ztrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, qry_work, &ineg_one, qry_rwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( lrwork_ );

    LAPACK_ztrevc3( &side_, &howmany_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &lwork_, &rwork[0], &lrwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.6.1

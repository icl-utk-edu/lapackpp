#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gehrd(
    int64_t n, int64_t ilo, int64_t ihi,
    float* A, int64_t lda,
    float* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gehrd(
    int64_t n, int64_t ilo, int64_t ihi,
    double* A, int64_t lda,
    double* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gehrd(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Reduces a general matrix A to upper Hessenberg form H by
/// an unitary similarity transformation: \f$ Q^H A Q = H \f$.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] ilo
///
/// @param[in] ihi
///     It is assumed that A is already upper triangular in rows
///     and columns 1:ilo-1 and ihi+1:n. ilo and ihi are normally
///     set by a previous call to `lapack::gebal`; otherwise they should be
///     set to 1 and n respectively. See Further Details.
///     - if n > 0, 1 <= ilo <= ihi <= n;
///     - if n = 0, ilo = 1 and ihi = 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the n-by-n general matrix to be reduced.
///     On exit, the upper triangle and the first subdiagonal of A
///     are overwritten with the upper Hessenberg matrix H, and the
///     elements below the first subdiagonal, with the array tau,
///     represent the unitary matrix Q as a product of elementary
///     reflectors. See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] tau
///     The vector tau of length n-1.
///     The scalar factors of the elementary reflectors (see Further
///     Details). Elements 1:ilo-1 and ihi:n-1 of tau are set to
///     zero.
///
/// @retval = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The matrix Q is represented as a product of (ihi-ilo) elementary
/// reflectors
///
///     \f[ Q = H(ilo) H(ilo+1) . . . H(ihi-1). \f]
///
/// Each H(i) has the form
///
///     \f[ H(i) = I - \tau v v^H \f]
///
/// where \f$ \tau \f$ is a scalar, and v is a vector with
/// v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
/// exit in A(i+2:ihi,i), and \f$ \tau \f$ in tau(i).
///
/// The contents of A are illustrated by the following example, with
/// n = 7, ilo = 2 and ihi = 6:
///
///     on entry,                        on exit,
///
///     ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
///     (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
///     (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
///     (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
///     (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
///     (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
///     (                         a )    (                          a )
///
/// where a denotes an element of the original matrix A, h denotes a
/// modified element of the upper Hessenberg matrix H, and vi denotes an
/// element of the vector defining H(i).
///
/// This routine is a slight modification of the LAPACK 3.0 `gehrd`
/// subroutine incorporating improvements proposed by Quintana-Orti and
/// Van de Geijn (2006). (See `lapack::lahr2`.)
///
/// @ingroup geev_computational
int64_t gehrd(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgehrd( &n_, &ilo_, &ihi_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

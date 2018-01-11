#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t ungbr(
    lapack::Vect vect, int64_t m, int64_t n, int64_t k,
    std::complex<float>* A, int64_t lda,
    std::complex<float> const* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cungbr( &vect_, &m_, &n_, &k_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cungbr( &vect_, &m_, &n_, &k_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Generates one of the complex unitary matrices \f$ Q \f$ or \f$ P^H \f$
/// determined by `lapack::gebrd` when reducing a complex matrix A to bidiagonal
/// form: \f$ A = Q B P^H. \f$ \f$ Q \f$ and \f$ P^H \f$ are defined as products of
/// elementary reflectors H(i) or G(i) respectively.
///
/// - If vect = Q, A is assumed to have been an m-by-k matrix,
///   and Q is of order m:
///   - if m >= k, \f$ Q = H(1) H(2) \dots H(k) \f$
///     and `ungbr` returns the first n columns of Q, where m >= n >= k;
///   - if m < k, \f$ Q = H(1) H(2) \dots H(m-1) \f$
///     and `ungbr` returns Q as an m-by-m matrix.
///
/// - If vect = P, A is assumed to have been a k-by-n matrix,
///   and \f$ P^H \f$ is of order n:
///   - if k < n, \f$ P^H = G(k) \dots G(2) G(1) \f$
///     and `ungbr` returns the first m rows of \f$ P^H, \f$ where n >= m >= k;
///   - if k >= n, \f$ P^H = G(n-1) \dots G(2) G(1) \f$
///     and `ungbr` returns \f$ P^H \f$ as an n-by-n matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::orgbr`.
///
/// @param[in] vect
///     Specifies whether the matrix \f$ Q \f$ or the matrix \f$ P^H \f$ is
///     required, as defined in the transformation applied by `lapack::gebrd`:
///     - lapack::Vect::Q: generate \f$ Q; \f$
///     - lapack::Vect::P: generate \f$ P^H, \f$
///
/// @param[in] m
///     The number of rows of the matrix Q or \f$ P^H \f$ to be returned.
///     m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix Q or \f$ P^H \f$ to be returned.
///     n >= 0.
///     - If vect = Q, m >= n >= min(m,k);
///     - if vect = P, n >= m >= min(n,k).
///
/// @param[in] k
///     - If vect = Q, the number of columns in the original m-by-k
///     matrix reduced by `lapack::gebrd`.
///     - If vect = P, the number of rows in the original k-by-n
///     matrix reduced by `lapack::gebrd`.
///     - k >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the vectors which define the elementary reflectors,
///     as returned by `lapack::gebrd`.
///     On exit, the m-by-n matrix \f$ Q \f$ or \f$ P^H, \f$
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= m.
///
/// @param[in] tau
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i) or G(i), which determines \f$ Q \f$ or \f$ P^H, \f$ as
///     returned by `lapack::gebrd` in its array argument tauq or taup.
///     - If vect = Q, the vector tau of length min(m,k);
///     - if vect = P, the vector tau of length min(n,k).
///
/// @retval = 0: successful exit
///
/// @ingroup gesvd_computational
int64_t ungbr(
    lapack::Vect vect, int64_t m, int64_t n, int64_t k,
    std::complex<double>* A, int64_t lda,
    std::complex<double> const* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zungbr( &vect_, &m_, &n_, &k_, A, &lda_, tau, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zungbr( &vect_, &m_, &n_, &k_, A, &lda_, tau, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

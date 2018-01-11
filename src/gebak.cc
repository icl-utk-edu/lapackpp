#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* scale, int64_t m,
    float* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_sgebak( &balance_, &side_, &n_, &ilo_, &ihi_, scale, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* scale, int64_t m,
    double* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_dgebak( &balance_, &side_, &n_, &ilo_, &ihi_, scale, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* scale, int64_t m,
    std::complex<float>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_cgebak( &balance_, &side_, &n_, &ilo_, &ihi_, scale, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Forms the right or left eigenvectors of a complex general
/// matrix by backward transformation on the computed eigenvectors of the
/// balanced matrix output by `lapack::gebal`.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] balance
///     Specifies the type of backward transformation required:
///     - lapack::Balance::None:    do nothing, return immediately;
///     - lapack::Balance::Permute: do backward transformation for permutation only;
///     - lapack::Balance::Scale:   do backward transformation for scaling only;
///     - lapack::Balance::Both:    do backward transformations for both
///                                 permutation and scaling.
///     balance must be the same as the argument balance supplied to `lapack::gebal`.
///
/// @param[in] side
///     - lapack::Side::Right: V contains right eigenvectors;
///     - lapack::Side::Left:  V contains left  eigenvectors.
///
/// @param[in] n
///     The number of rows of the matrix V. n >= 0.
///
/// @param[in] ilo
///
/// @param[in] ihi
///     The integers ilo and ihi determined by `lapack::gebal`.
///     - If n > 0, then 1 <= ilo <= ihi <= n;
///     - if n = 0, then ilo=1 and ihi=0.
///
/// @param[in] scale
///     The vector scale of length n.
///     Details of the permutation and scaling factors, as returned
///     by `lapack::gebal`.
///
/// @param[in] m
///     The number of columns of the matrix V. m >= 0.
///
/// @param[in,out] V
///     The n-by-m matrix V, stored in an ldv-by-m array.
///     On entry, the matrix of right or left eigenvectors to be
///     transformed, as returned by `lapack::hsein` or `lapack::trevc`.
///     On exit, V is overwritten by the transformed eigenvectors.
///
/// @param[in] ldv
///     The leading dimension of the array V. ldv >= max(1,n).
///
/// @retval = 0: successful exit
///
/// @ingroup geev_computational
int64_t gebak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* scale, int64_t m,
    std::complex<double>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_zgebak( &balance_, &side_, &n_, &ilo_, &ihi_, scale, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

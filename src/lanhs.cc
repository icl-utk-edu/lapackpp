#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup norm
float lanhs(
    lapack::Norm norm, int64_t n,
    float const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    std::vector< float > work( max(1,lwork) );

    return LAPACK_slanhs( &norm_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lanhs(
    lapack::Norm norm, int64_t n,
    double const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    std::vector< double > work( max(1,lwork) );

    return LAPACK_dlanhs( &norm_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lanhs(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    std::vector< float > work( max(1,lwork) );

    return LAPACK_clanhs( &norm_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// Hessenberg matrix A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] norm
///     The value to be returned:
///     - lapack::Norm::Max: max norm: max(abs(A(i,j))).
///                          Note this is not a consistent matrix norm.
///     - lapack::Norm::One: one norm: maximum column sum
///     - lapack::Norm::Inf: infinity norm: maximum row sum
///     - lapack::Norm::Fro: Frobenius norm: square root of sum of squares
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The n-by-n upper Hessenberg matrix A; the part of A below the
///     first sub-diagonal is not referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(n,1).
///
/// @ingroup norm
double lanhs(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    std::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhs( &norm_, &n_, A, &lda_, &work[0] );
}

}  // namespace lapack

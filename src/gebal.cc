#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebal(
    lapack::Balance balance, int64_t n,
    float* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    float* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_sgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, scale, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebal(
    lapack::Balance balance, int64_t n,
    double* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    double* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_dgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, scale, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t gebal(
    lapack::Balance balance, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    float* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_cgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, scale, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Balances a general complex matrix A. This involves, first,
/// permuting A by a similarity transformation to isolate eigenvalues
/// in the first 1 to ilo-1 and last ihi+1 to n elements on the
/// diagonal; and second, applying a diagonal similarity transformation
/// to rows and columns ilo to ihi to make the rows and columns as
/// close in norm as possible. Both steps are optional.
///
/// Balancing may reduce the 1-norm of the matrix, and improve the
/// accuracy of the computed eigenvalues and/or eigenvectors.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] balance
///     Specifies the operations to be performed on A:
///     - lapack::Balance::None: none: simply set ilo = 1, ihi = n, scale(I) = 1.0
///         for i = 1, ..., n;
///     - lapack::Balance::Permute: permute only;
///     - lapack::Balance::Scale: scale only;
///     - lapack::Balance::Both: both permute and scale.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the input matrix A.
///     On exit, A is overwritten by the balanced matrix.
///     If balance = None, A is not referenced.
///     See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] ilo
///
/// @param[out] ihi
///     ilo and ihi are set to integers such that on exit
///     A(i,j) = 0 if i > j and j = 1, ..., ilo-1 or i = ihi+1, ..., n.
///     If balance = None or Scale, ilo = 1 and ihi = n.
///
/// @param[out] scale
///     The vector scale of length n.
///     Details of the permutations and scaling factors applied to
///     A. If P(j) is the index of the row and column interchanged
///     with row and column j and D(j) is the scaling factor
///     applied to row and column j, then:     \n
///     scale(j) = P(j) for j = 1, ..., ilo-1; \n
///     scale(j) = D(j) for j = ilo, ..., ihi; \n
///     scale(j) = P(j) for j = ihi+1, ..., n. \n
///     The order in which the interchanges are made is n to ihi+1,
///     then 1 to ilo-1.
///
/// @retval = 0: successful exit.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The permutations consist of row and column interchanges which put
/// the matrix in the form
/**
    \f[ P A P = \left[ \begin{array}{ccc}
                    T1  &  X  &  Y   \\
                    0   &  B  &  Z   \\
                    0   &  0  &  T2  \\
                \end{array} \right], \f]
*/
///
/// where T1 and T2 are upper triangular matrices whose eigenvalues lie
/// along the diagonal.  The column indices ilo and ihi mark the starting
/// and ending columns of the submatrix B. Balancing consists of applying
/// a diagonal similarity transformation \f$ D^{-1} B D \f$ to make the
/// 1-norms of each row of B and its corresponding column nearly equal.
/// The output matrix is
/**
    \f[ \left[ \begin{array}{ccc}
            T1  &  X D         &  Y         \\
            0   &  D^{-1} B D  &  D^{-1} Z  \\
            0   &  0           &  T2        \\
        \end{array} \right]. \f]
*/
///
/// Information about the permutations P and the diagonal matrix D is
/// returned in the vector scale.
///
/// This subroutine is based on the EISPACK routine CBAL.
///
/// Modified by Tzu-Yi Chen, Computer Science Division, University of
/// California at Berkeley, USA
///
/// @ingroup geev_computational
int64_t gebal(
    lapack::Balance balance, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    double* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_zgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, scale, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

}  // namespace lapack

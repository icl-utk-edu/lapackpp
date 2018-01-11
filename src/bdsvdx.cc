#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 6  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup bdsvd
int64_t bdsvdx(
    lapack::Uplo uplo, lapack::Job jobz, lapack::Range range, int64_t n,
    float const* D,
    float const* E, float vl, float vu, int64_t il, int64_t iu,
    int64_t* nfound,
    float* S,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int nfound_ = (blas_int) *nfound;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (14*n) );
    std::vector< blas_int > iwork( (12*n) );

    LAPACK_sbdsvdx( &uplo_, &jobz_, &range_, &n_, D, E, &vl, &vu, &il_, &iu_, &nfound_, S, Z, &ldz_, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the singular value decomposition (SVD) of a real
/// n-by-n (upper or lower) bidiagonal matrix B, \f$ B = U S VT \f$,
/// where S is a diagonal matrix with non-negative diagonal elements
/// (the singular values of B), and U and VT are orthogonal matrices
/// of left and right singular vectors, respectively.
///
/// Given an upper bidiagonal B with diagonal \f$ D = [ d_1 d_2 ... d_n ] \f$
/// and superdiagonal \f$ E = [ e_1 e_2 ... e_{n-1} ] \f$, `bdsvdx` computes the
/// singular value decompositon of B through the eigenvalues and
/// eigenvectors of the 2n-by-2n tridiagonal matrix
/**
    \f[
    TGK = \left[ \begin{array}{cccccc}
        0    &  d_1                              \\
        d_1  &  0    &  e_1                      \\
             &  e_1  &  0    &  d_2              \\
             &       &  d_2  &  .    &  .        \\
             &       &       &  .    &  .  &  .  \\
          \end{array} \right]
    \f]
*/
///
/// If (s,u,v) is a singular triplet of B with ||u|| = ||v|| = 1, then
/// (+/-s,q), ||q|| = 1, are eigenpairs of TGK, with q = P * (u' +/- v') /
/// sqrt(2) = ( v_1 u_1 v_2 u_2 ... v_n u_n ) / sqrt(2), and
/// P = [ e_{n+1} e_{1} e_{n+2} e_{2} ... ].
///
/// Given a TGK matrix, one can either a) compute -s,-v and change signs
/// so that the singular values (and corresponding vectors) are already in
/// descending order (as in `lapack::gesvd`/`lapack::gesdd`) or b) compute s,v and reorder
/// the values (and corresponding vectors). `bdsvdx` implements a) by
/// calling `lapack::stevx` (bisection plus inverse iteration, to be replaced
/// with a version of the Multiple Relative Robust Representation
/// algorithm. (See P. Willems and B. Lang, A framework for the MR^3
/// algorithm: theory and implementation, SIAM J. Sci. Comput.,
/// 35:740-766, 2013.)
///
/// Overloaded versions are available for
/// `float`, `double`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: B is upper bidiagonal;
///     - lapack::Uplo::Lower: B is lower bidiagonal.
///
/// @param[in] jobz
///     - lapack::Job::NoVec: Compute singular values only;
///     - lapack::Job::Vec:   Compute singular values and singular vectors.
///
/// @param[in] range
///     - lapack::Range::All:
///             all singular values will be found.
///     - lapack::Range::Value:
///             all singular values in the half-open interval [vl,vu) will be found.
///     - lapack::Range::Index:
///             the il-th through iu-th singular values will be found.
///
/// @param[in] n
///     The order of the bidiagonal matrix. n >= 0.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of the bidiagonal matrix B.
///
/// @param[in] E
///     The vector E of length max(1,n-1).
///     The (n-1) superdiagonal elements of the bidiagonal matrix
///     B in elements 1 to n-1.
///
/// @param[in] vl
///     If range=Value, the lower bound of the interval to
///     be searched for singular values. vu > vl.
///     Not referenced if range = All or Index.
///
/// @param[in] vu
///     If range=Value, the upper bound of the interval to
///     be searched for singular values. vu > vl.
///     Not referenced if range = All or Index.
///
/// @param[in] il
///     If range=Index, the index of the
///     smallest singular value to be returned.
///     1 <= il <= iu <= min(M,n), if min(M,n) > 0.
///     Not referenced if range = All or Value.
///
/// @param[in] iu
///     If range=Index, the index of the
///     largest singular value to be returned.
///     1 <= il <= iu <= min(M,n), if min(M,n) > 0.
///     Not referenced if range = All or Value.
///
/// @param[out] nfound
///     The total number of singular values found. 0 <= nfound <= n.
///     - If range = All, nfound = n;
///     - if range = Index, nfound = iu-il+1.
///
/// @param[out] S
///     The vector S of length n.
///     The first nfound elements contain the selected singular values in
///     ascending order.
///
/// @param[out] Z
///     The (2*n)-by-zcol matrix Z, stored in an (2*n)-by-zcol array.
///     - If jobz = Vec, then if successful the first nfound columns of Z
///     contain the singular vectors of the matrix B corresponding to
///     the selected singular values, with U in rows 1 to n and V
///     in rows n+1 to 2*n, i.e.
/**
        \f[
        Z = \left[ \begin{array}{c}
            U  \\
            V  \\
        \end{array} \right]
        \f]
*/
///     - If jobz = NoVec, then Z is not referenced.
///     \n
///     Note: The user must ensure that zcol >= nfound+1 columns are
///     supplied in the array Z; if range = Value, the exact value of
///     nfound is not known in advance and an upper bound must be used.
///
/// @param[in] ldz
///     The leading dimension of the array Z. ldz >= 1, and if
///     jobz = Vec, ldz >= max(2,2*n).
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, then i eigenvectors failed to converge
///              in `lapack::stevx`. The indices of the eigenvectors
///              (as returned by `lapack::stevx`) are stored in the
///              array iwork.
/// @retval > n: if return value = 2*n + 1, an internal error occurred.
///
/// @ingroup bdsvd
int64_t bdsvdx(
    lapack::Uplo uplo, lapack::Job jobz, lapack::Range range, int64_t n,
    double const* D,
    double const* E, double vl, double vu, int64_t il, int64_t iu,
    int64_t* nfound,
    double* S,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int nfound_ = (blas_int) *nfound;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (14*n) );
    std::vector< blas_int > iwork( (12*n) );

    LAPACK_dbdsvdx( &uplo_, &jobz_, &range_, &n_, D, E, &vl, &vu, &il_, &iu_, &nfound_, S, Z, &ldz_, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6

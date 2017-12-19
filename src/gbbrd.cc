#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    float* AB, int64_t ldab,
    float* D,
    float* E,
    float* Q, int64_t ldq,
    float* PT, int64_t ldpt,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldpt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ncc_ = (blas_int) ncc;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldpt_ = (blas_int) ldpt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (2*max(m,n)) );

    LAPACK_sgbbrd( &vect_, &m_, &n_, &ncc_, &kl_, &ku_, AB, &ldab_, D, E, Q, &ldq_, PT, &ldpt_, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    double* AB, int64_t ldab,
    double* D,
    double* E,
    double* Q, int64_t ldq,
    double* PT, int64_t ldpt,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldpt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ncc_ = (blas_int) ncc;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldpt_ = (blas_int) ldpt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (2*max(m,n)) );

    LAPACK_dgbbrd( &vect_, &m_, &n_, &ncc_, &kl_, &ku_, AB, &ldab_, D, E, Q, &ldq_, PT, &ldpt_, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    std::complex<float>* AB, int64_t ldab,
    float* D,
    float* E,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* PT, int64_t ldpt,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldpt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ncc_ = (blas_int) ncc;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldpt_ = (blas_int) ldpt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (max(m,n)) );
    std::vector< float > rwork( (max(m,n)) );

    LAPACK_cgbbrd( &vect_, &m_, &n_, &ncc_, &kl_, &ku_, AB, &ldab_, D, E, Q, &ldq_, PT, &ldpt_, C, &ldc_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Reduces a general m-by-n band matrix A to real upper
/// bidiagonal form B by a unitary transformation: \f$ Q^H A P = B \f$.
///
/// The routine computes B, and optionally forms \f$ Q \f$ or \f$ P^H \f$, or computes
/// \f$ Q^H C \f$ for a given matrix C.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] vect
///     Whether or not the matrices Q and P^H are to be
///     formed.
///     - lapack::Vect::None: do not form \f$ Q \f$ or \f$ P^H \f$;
///     - lapack::Vect::Q:    form \f$ Q   \f$ only;
///     - lapack::Vect::P:    form \f$ P^H \f$ only;
///     - lapack::Vect::Both: form both.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] ncc
///     The number of columns of the matrix C. ncc >= 0.
///
/// @param[in] kl
///     The number of subdiagonals of the matrix A. kl >= 0.
///
/// @param[in] ku
///     The number of superdiagonals of the matrix A. ku >= 0.
///
/// @param[in,out] AB
///     The m-by-n band matrix AB, stored in an ldab-by-n array.
///     On entry, the m-by-n band matrix A, stored in rows 1 to
///     kl+ku+1. The j-th column of A is stored in the j-th column of
///     the array AB as follows:
///     AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku) <= i <= min(m,j+kl).
///     On exit, A is overwritten by values generated during the
///     reduction.
///
/// @param[in] ldab
///     The leading dimension of the array A. ldab >= kl+ku+1.
///
/// @param[out] D
///     The vector D of length min(m,n).
///     The diagonal elements of the bidiagonal matrix B.
///
/// @param[out] E
///     The vector E of length min(m,n)-1.
///     The superdiagonal elements of the bidiagonal matrix B.
///
/// @param[out] Q
///     The m-by-m matrix Q, stored in an ldq-by-m array.
///     - If vect = Q or Both, the m-by-m unitary matrix Q.
///     - If vect = None or P, the array Q is not referenced.
///
/// @param[in] ldq
///     The leading dimension of the array Q.
///     - If vect = Q or Both, ldq >= max(1,m);
///     - otherwise, ldq >= 1.
///
/// @param[out] PT
///     The n-by-n matrix PT, stored in an ldpt-by-n array.
///     - If vect = P or Both, the n-by-n unitary matrix \f$ P^H \f$;
///     - If vect = None or Q, the array PT is not referenced.
///
/// @param[in] ldpt
///     The leading dimension of the array PT.
///     - If vect = P or Both, ldpt >= max(1,n);
///     - otherwise, ldpt >= 1.
///
/// @param[in,out] C
///     The m-by-ncc matrix C, stored in an ldc-by-ncc array.
///     On entry, an m-by-ncc matrix C.
///     On exit, C is overwritten by \f$ Q^H C \f$.
///     C is not referenced if ncc = 0.
///
/// @param[in] ldc
///     The leading dimension of the array C.
///     - If ncc > 0, ldc >= max(1,m);
///     - if ncc = 0, ldc >= 1.
///
/// @retval = 0: successful exit.
///
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    std::complex<double>* AB, int64_t ldab,
    double* D,
    double* E,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* PT, int64_t ldpt,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ncc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldpt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ncc_ = (blas_int) ncc;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldpt_ = (blas_int) ldpt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (max(m,n)) );
    std::vector< double > rwork( (max(m,n)) );

    LAPACK_zgbbrd( &vect_, &m_, &n_, &ncc_, &kl_, &ku_, AB, &ldab_, D, E, Q, &ldq_, PT, &ldpt_, C, &ldc_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

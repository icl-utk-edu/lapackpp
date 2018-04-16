#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    float const* V, int64_t ldv,
    float const* T, int64_t ldt,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    blas_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< float > work( ldwork_ * k );

    LAPACK_slarfb(
        &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    double const* V, int64_t ldv,
    double const* T, int64_t ldt,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    blas_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< double > work( ldwork_ * k );

    LAPACK_dlarfb(
        &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* T, int64_t ldt,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    blas_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<float> > work( ldwork_ * k );

    LAPACK_clarfb(
        &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
/// Applies a block reflector \f$ H \f$ or its transpose \f$ H^H \f$ to a
/// m-by-n matrix C, from either the left or the right.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply \f$ H \f$ or \f$ H^H \f$ from the Left
///     - lapack::Side::Right: apply \f$ H \f$ or \f$ H^H \f$ from the Right
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   apply \f$ H   \f$ (No transpose)
///     - lapack::Op::ConjTrans: apply \f$ H^H \f$ (Conjugate transpose)
///
/// @param[in] direct
///     Indicates how H is formed from a product of elementary
///     reflectors
///     - lapack::Direct::Forward:  \f$ H = H(1) H(2) \dots H(k) \f$
///     - lapack::Direct::Backward: \f$ H = H(k) \dots H(2) H(1) \f$
///
/// @param[in] storev
///     Indicates how the vectors which define the elementary
///     reflectors are stored:
///     - lapack::StoreV::Columnwise
///     - lapack::StoreV::Rowwise
///
/// @param[in] m
///     The number of rows of the matrix C.
///
/// @param[in] n
///     The number of columns of the matrix C.
///
/// @param[in] k
///     The order of the matrix T (= the number of elementary
///     reflectors whose product defines the block reflector).
///     - If side = Left,  m >= k >= 0;
///     - if side = Right, n >= k >= 0.
///
/// @param[in] V
///     - If storev = Columnwise:
///       - if side = Left,  the m-by-k matrix V, stored in an ldv-by-k array;
///       - if side = Right, the n-by-k matrix V, stored in an ldv-by-k array.
///     - If storev = Rowwise:
///       - if side = Left,  the k-by-m matrix V, stored in an ldv-by-m array;
///       - if side = Right, the k-by-n matrix V, stored in an ldv-by-n array.
///     - See Further Details.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     - If storev = Columnwise and side = Left,  ldv >= max(1,m);
///     - if storev = Columnwise and side = Right, ldv >= max(1,n);
///     - if storev = Rowwise, ldv >= k.
///
/// @param[in] T
///     The k-by-k matrix T, stored in an ldt-by-k array.
///     The triangular k-by-k matrix T in the representation of the
///     block reflector.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= k.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by
///     \f$ H C \f$ or \f$ H^H C \f$ or \f$ C H \f$ or \f$ C H^H \f$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The shape of the matrix V and the storage of the vectors which define
/// the H(i) is best illustrated by the following example with n = 5 and
/// k = 3. The elements equal to 1 are not stored. The rest of the
/// array is not used.
///
///     direct = Forward and             direct = Forward and
///     storev = Columnwise:             storev = Rowwise:
///
///     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
///         ( v1  1    )                     (     1 v2 v2 v2 )
///         ( v1 v2  1 )                     (        1 v3 v3 )
///         ( v1 v2 v3 )
///         ( v1 v2 v3 )
///
///     direct = Backward and            direct = Backward and
///     storev = Columnwise:             storev = Rowwise:
///
///     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
///         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
///         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
///         (     1 v3 )
///         (        1 )
///
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* T, int64_t ldt,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    blas_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<double> > work( ldwork_ * k );

    LAPACK_zlarfb(
        &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &ldwork_ );
}

}  // namespace lapack

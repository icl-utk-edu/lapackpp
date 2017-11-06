#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t trevc(
    lapack::Side side, lapack::HowMany howmny,
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
    char side_ = side2char( side );
    char howmny_ = howmany2char( howmny );
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

    // allocate workspace
    std::vector< float > work( (3*n) );

    LAPACK_strevc( &side_, &howmny_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &info_ );
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
int64_t trevc(
    lapack::Side side, lapack::HowMany howmny,
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
    char side_ = side2char( side );
    char howmny_ = howmany2char( howmny );
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

    // allocate workspace
    std::vector< double > work( (3*n) );

    LAPACK_dtrevc( &side_, &howmny_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &info_ );
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
int64_t trevc(
    lapack::Side side, lapack::HowMany howmny,
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
    char side_ = side2char( side );
    char howmny_ = howmany2char( howmny );
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

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_ctrevc( &side_, &howmny_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trevc(
    lapack::Side side, lapack::HowMany howmny,
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
    char side_ = side2char( side );
    char howmny_ = howmany2char( howmny );
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

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_ztrevc( &side_, &howmny_, select_ptr, &n_, T, &ldt_, VL, &ldvl_, VR, &ldvr_, &mm_, &m_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

}  // namespace lapack

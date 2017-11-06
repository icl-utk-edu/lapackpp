#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
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

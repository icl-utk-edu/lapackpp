#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hbgvx(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<float>* AB, int64_t ldab,
    std::complex<float>* BB, int64_t ldbb,
    std::complex<float>* Q, int64_t ldq, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* m,
    float* W,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(il) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int m_ = (blas_int) *m;
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (n) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (n) );
    std::vector< float > rwork( (7*n) );
    std::vector< blas_int > iwork( (5*n) );

    LAPACK_chbgvx( &jobz_, &range_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &m_, W, Z, &ldz_, &work[0], &rwork[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hbgvx(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<double>* AB, int64_t ldab,
    std::complex<double>* BB, int64_t ldbb,
    std::complex<double>* Q, int64_t ldq, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* m,
    double* W,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(il) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int m_ = (blas_int) *m;
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (n) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (n) );
    std::vector< double > rwork( (7*n) );
    std::vector< blas_int > iwork( (5*n) );

    LAPACK_zhbgvx( &jobz_, &range_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &m_, W, Z, &ldz_, &work[0], &rwork[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

}  // namespace lapack

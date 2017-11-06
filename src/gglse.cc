#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* C,
    float* D,
    float* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* C,
    double* D,
    double* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* C,
    std::complex<float>* D,
    std::complex<float>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* C,
    std::complex<double>* D,
    std::complex<double>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

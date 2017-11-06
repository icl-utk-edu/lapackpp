#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tgsyl(
    lapack::Op trans, int64_t ijob, int64_t m, int64_t n,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float* C, int64_t ldc,
    float const* D, int64_t ldd,
    float const* E, int64_t lde,
    float* F, int64_t ldf,
    float* dif,
    float* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(ijob) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lde) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldf) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int ijob_ = (blas_int) ijob;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldd_ = (blas_int) ldd;
    blas_int lde_ = (blas_int) lde;
    blas_int ldf_ = (blas_int) ldf;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_stgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( (m+n+6) );

    LAPACK_stgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsyl(
    lapack::Op trans, int64_t ijob, int64_t m, int64_t n,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double* C, int64_t ldc,
    double const* D, int64_t ldd,
    double const* E, int64_t lde,
    double* F, int64_t ldf,
    double* dif,
    double* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(ijob) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lde) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldf) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int ijob_ = (blas_int) ijob;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldd_ = (blas_int) ldd;
    blas_int lde_ = (blas_int) lde;
    blas_int ldf_ = (blas_int) ldf;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dtgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( (m+n+6) );

    LAPACK_dtgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsyl(
    lapack::Op trans, int64_t ijob, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* C, int64_t ldc,
    std::complex<float> const* D, int64_t ldd,
    std::complex<float> const* E, int64_t lde,
    std::complex<float>* F, int64_t ldf,
    float* dif,
    float* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(ijob) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lde) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldf) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int ijob_ = (blas_int) ijob;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldd_ = (blas_int) ldd;
    blas_int lde_ = (blas_int) lde;
    blas_int ldf_ = (blas_int) ldf;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_ctgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< blas_int > iwork( (m+n+2) );

    LAPACK_ctgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tgsyl(
    lapack::Op trans, int64_t ijob, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* C, int64_t ldc,
    std::complex<double> const* D, int64_t ldd,
    std::complex<double> const* E, int64_t lde,
    std::complex<double>* F, int64_t ldf,
    double* dif,
    double* scale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(ijob) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lde) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldf) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int ijob_ = (blas_int) ijob;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldd_ = (blas_int) ldd;
    blas_int lde_ = (blas_int) lde;
    blas_int ldf_ = (blas_int) ldf;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_ztgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< blas_int > iwork( (m+n+2) );

    LAPACK_ztgsyl( &trans_, &ijob_, &m_, &n_, A, &lda_, B, &ldb_, C, &ldc_, D, &ldd_, E, &lde_, F, &ldf_, dif, scale, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

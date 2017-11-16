#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Job job, lapack::CompQ compq,
    bool const* select, int64_t n,
    float* T, int64_t ldt,
    float* Q, int64_t ldq,
    float* WR,
    float* WI,
    int64_t* m,
    float* s,
    float* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_strsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, WR, WI, &m_, s, sep, qry_work, &ineg_one, qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_strsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, WR, WI, &m_, s, sep, &work[0], &lwork_, &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Job job, lapack::CompQ compq,
    bool const* select, int64_t n,
    double* T, int64_t ldt,
    double* Q, int64_t ldq,
    double* WR,
    double* WI,
    int64_t* m,
    double* s,
    double* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dtrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, WR, WI, &m_, s, sep, qry_work, &ineg_one, qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_dtrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, WR, WI, &m_, s, sep, &work[0], &lwork_, &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Job job, lapack::CompQ compq,
    bool const* select, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* W,
    int64_t* m,
    float* s,
    float* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_ctrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, W, &m_, s, sep, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_ctrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, W, &m_, s, sep, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Job job, lapack::CompQ compq,
    bool const* select, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* W,
    int64_t* m,
    double* s,
    double* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    #if 1
        // 32-bit copy
        std::vector< blas_int > select_( &select[0], &select[(n)] );
        blas_int const* select_ptr = &select_[0];
    #else
        blas_int const* select_ptr = select_;
    #endif
    blas_int n_ = (blas_int) n;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int m_ = (blas_int) *m;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_ztrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, W, &m_, s, sep, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_ztrsen( &job_, &compq_, select_ptr, &n_, T, &ldt_, Q, &ldq_, W, &m_, s, sep, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

}  // namespace lapack

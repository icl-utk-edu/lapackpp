#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Sense sense, lapack::Job compq,
    bool const* select, int64_t n,
    float* T, int64_t ldt,
    float* Q, int64_t ldq,
    std::complex<float>* W,
    int64_t* m,
    float* s,
    float* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char sense_ = sense2char( sense );
    char compq_ = job_comp2char( compq );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // split-complex representation
    std::vector< float > WR( max( 1, n ) );
    std::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_strsen(
        &sense_, &compq_,
        select_ptr, &n_,
        T, &ldt_,
        Q, &ldq_,
        &WR[0],
        &WI[0],
        &m_, s, sep,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< lapack_int > iwork( liwork_ );

    LAPACK_strsen(
        &sense_, &compq_,
        select_ptr, &n_,
        T, &ldt_,
        Q, &ldq_,
        &WR[0],
        &WI[0],
        &m_, s, sep,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Sense sense, lapack::Job compq,
    bool const* select, int64_t n,
    double* T, int64_t ldt,
    double* Q, int64_t ldq,
    std::complex<double>* W,
    int64_t* m,
    double* s,
    double* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char sense_ = sense2char( sense );
    char compq_ = job_comp2char( compq );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // split-complex representation
    std::vector< double > WR( max( 1, n ) );
    std::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dtrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        T, &ldt_,
        Q, &ldq_,
        &WR[0],
        &WI[0],
        &m_, s, sep,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< lapack_int > iwork( liwork_ );

    LAPACK_dtrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        T, &ldt_,
        Q, &ldq_,
        &WR[0],
        &WI[0],
        &m_, s, sep,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Sense sense, lapack::Job compq,
    bool const* select, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* W,
    int64_t* m,
    float* s,
    float* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char sense_ = sense2char( sense );
    char compq_ = job_comp2char( compq );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_ctrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) W, &m_, s, sep,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_ctrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) W, &m_, s, sep,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trsen(
    lapack::Sense sense, lapack::Job compq,
    bool const* select, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* W,
    int64_t* m,
    double* s,
    double* sep )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char sense_ = sense2char( sense );
    char compq_ = job_comp2char( compq );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_ztrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) W, &m_, s, sep,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_ztrsen(
        &sense_, &compq_,
        select_ptr, &n_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) W, &m_, s, sep,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

}  // namespace lapack

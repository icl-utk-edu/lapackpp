#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t bdsdc(
    lapack::Uplo uplo, lapack::CompQ compq, int64_t n,
    float* D,
    float* E,
    float* U, int64_t ldu,
    float* VT, int64_t ldvt,
    float* Q,
    int64_t* IQ )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // IQ disabled for now, due to complicated dimension
    blas_int IQ_[1];
    blas_int *IQ_ptr = &IQ_[0];

    // formulas from docs
    int64_t lwork;
    switch (compq) {
        case CompQ::NoVec:      lwork = 4*n; break;
        case CompQ::Vec:        lwork = 6*n; break;
        case CompQ::CompactVec: lwork = 3*n*n + 4*n; break;
        case CompQ::Update:     assert( false ); break;
    }

    // allocate workspace
    std::vector< float > work( (max( (int64_t) 1, lwork)) );
    std::vector< blas_int > iwork( (8*n) );

    LAPACK_sbdsdc( &uplo_, &compq_, &n_, D, E, U, &ldu_, VT, &ldvt_, Q, IQ_ptr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bdsdc(
    lapack::Uplo uplo, lapack::CompQ compq, int64_t n,
    double* D,
    double* E,
    double* U, int64_t ldu,
    double* VT, int64_t ldvt,
    double* Q,
    int64_t* IQ )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char compq_ = compq2char( compq );
    blas_int n_ = (blas_int) n;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // IQ disabled for now, due to complicated dimension
    blas_int IQ_[1];
    blas_int *IQ_ptr = &IQ_[0];

    // formulas from docs
    int64_t lwork;
    switch (compq) {
        case CompQ::NoVec:      lwork = 4*n; break;
        case CompQ::Vec:        lwork = 6*n; break;
        case CompQ::CompactVec: lwork = 3*n*n + 4*n; break;
        case CompQ::Update:     assert( false ); break;
    }

    // allocate workspace
    std::vector< double > work( (max( (int64_t) 1, lwork)) );
    std::vector< blas_int > iwork( (8*n) );

    LAPACK_dbdsdc( &uplo_, &compq_, &n_, D, E, U, &ldu_, VT, &ldvt_, Q, IQ_ptr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

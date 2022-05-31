// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    int64_t* ipiv,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int lda_ = (lapack_int) lda;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    // MKL (seen in 2019-2021) has bug in sysv_aa query;
    // query sytrf_aa and sytrs_aa instead.
    #ifdef BLAS_HAVE_MKL
        LAPACK_ssytrf_aa(
            &uplo_, &n_,
            A, &lda_,
            ipiv_ptr,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
        LAPACK_ssytrs_aa(
            &uplo_, &n_, &nrhs_,
            A, &lda_,
            ipiv_ptr,
            B, &ldb_,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lwork_ = max( lwork_, real(qry_work[0]) );
    #else
        LAPACK_ssysv_aa(
            &uplo_, &n_, &nrhs_,
            A, &lda_,
            ipiv_ptr,
            B, &ldb_,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
    #endif

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_ssysv_aa(
        &uplo_, &n_, &nrhs_,
        A, &lda_,
        ipiv_ptr,
        B, &ldb_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    int64_t* ipiv,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int lda_ = (lapack_int) lda;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    // MKL (seen in 2019-2021) has bug in sysv_aa query;
    // query sytrf_aa and sytrs_aa instead.
    #ifdef BLAS_HAVE_MKL
        LAPACK_dsytrf_aa(
            &uplo_, &n_,
            A, &lda_,
            ipiv_ptr,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
        LAPACK_dsytrs_aa(
            &uplo_, &n_, &nrhs_,
            A, &lda_,
            ipiv_ptr,
            B, &ldb_,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lwork_ = max( lwork_, real(qry_work[0]) );
    #else
        LAPACK_dsysv_aa(
            &uplo_, &n_, &nrhs_,
            A, &lda_,
            ipiv_ptr,
            B, &ldb_,
            qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
    #endif


    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dsysv_aa(
        &uplo_, &n_, &nrhs_,
        A, &lda_,
        ipiv_ptr,
        B, &ldb_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int lda_ = (lapack_int) lda;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    // MKL (seen in 2019-2021) has bug in sysv_aa query;
    // query sytrf_aa and sytrs_aa instead.
    #ifdef BLAS_HAVE_MKL
        LAPACK_csytrf_aa(
            &uplo_, &n_,
            (lapack_complex_float*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_float*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
        LAPACK_csytrs_aa(
            &uplo_, &n_, &nrhs_,
            (lapack_complex_float*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_float*) B, &ldb_,
            (lapack_complex_float*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lwork_ = max( lwork_, real(qry_work[0]) );
    #else
        LAPACK_csysv_aa(
            &uplo_, &n_, &nrhs_,
            (lapack_complex_float*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_float*) B, &ldb_,
            (lapack_complex_float*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
    #endif

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_csysv_aa(
        &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        ipiv_ptr,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the solution to a system of linear equations
///     $A X = B$,
/// where A is an n-by-n symmetric matrix and X and B are n-by-nrhs
/// matrices.
///
/// Aasen's algorithm is used to factor A as
///     $A = U T U^T$ if uplo = Upper, or
///     $A = L T L^T$ if uplo = Lower,
/// where U (or L) is a product of permutation and unit upper (lower)
/// triangular matrices, and T is symmetric tridiagonal. The factored
/// form of A is then used to solve the system of equations $A X = B$.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::hesv_aa` is an alias for this.
/// For complex Hermitian matrices, see `lapack::hesv_aa`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the symmetric matrix A.
///     - If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced.
///
///     - If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
///     - On successful exit, the tridiagonal matrix T and the
///     multipliers used to obtain the factor U or L from the
///     factorization $A = U T U^T$ or $A = L T L^T$ as computed by
///     `lapack::sytrf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     On exit, it contains the details of the interchanges, i.e.,
///     the row and column k of A were interchanged with the
///     row and column ipiv(k).
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On successful exit, the n-by-nrhs solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, D(i,i) is exactly zero. The factorization
///              has been completed, but the block diagonal matrix D is
///              exactly singular, so the solution could not be computed.
///
/// @ingroup sysv
int64_t sysv_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int lda_ = (lapack_int) lda;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    // MKL (seen in 2019-2021) has bug in sysv_aa query;
    // query sytrf_aa and sytrs_aa instead.
    #ifdef BLAS_HAVE_MKL
        LAPACK_zsytrf_aa(
            &uplo_, &n_,
            (lapack_complex_double*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_double*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
        LAPACK_zsytrs_aa(
            &uplo_, &n_, &nrhs_,
            (lapack_complex_double*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_double*) B, &ldb_,
            (lapack_complex_double*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lwork_ = max( lwork_, real(qry_work[0]) );
    #else
        LAPACK_zsysv_aa(
            &uplo_, &n_, &nrhs_,
            (lapack_complex_double*) A, &lda_,
            ipiv_ptr,
            (lapack_complex_double*) B, &ldb_,
            (lapack_complex_double*) qry_work, &ineg_one, &info_
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
        );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real(qry_work[0]);
    #endif

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zsysv_aa(
        &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        ipiv_ptr,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7

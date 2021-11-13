#ifndef LAPACK_BLAS_WRAPPERS_HH
#define LAPACK_BLAS_WRAPPERS_HH

#include "blas/fortran.h"

// todo: put into BLAS++

namespace blas {

//==============================================================================
// hbmv

// -----------------------------------------------------------------------------
extern "C" {

#define BLAS_ssbmv BLAS_FORTRAN_NAME( ssbmv, SSBMV )
void BLAS_ssbmv(
    char const *uplo,
    blas_int const *n, blas_int const *kd,
    float const *alpha,
    float const *AB, blas_int const *ldab,
    float const *B,  blas_int const *incb,
    float const *beta,
    float       *C,  blas_int const *incc );

#define BLAS_dsbmv BLAS_FORTRAN_NAME( dsbmv, DSBMV )
void BLAS_dsbmv(
    char const *uplo,
    blas_int const *n, blas_int const *kd,
    double const *alpha,
    double const *AB, blas_int const *ldab,
    double const *B,  blas_int const *incb,
    double const *beta,
    double       *C,  blas_int const *incc );

#define BLAS_chbmv BLAS_FORTRAN_NAME( chbmv, CHBMV )
void BLAS_chbmv(
    char const *uplo,
    blas_int const *n, blas_int const *kd,
    blas_complex_float const *alpha,
    blas_complex_float const *AB, blas_int const *ldab,
    blas_complex_float const *B,  blas_int const *incb,
    blas_complex_float const *beta,
    blas_complex_float       *C,  blas_int const *incc );

#define BLAS_zhbmv BLAS_FORTRAN_NAME( zhbmv, ZHBMV )
void BLAS_zhbmv(
    char const *uplo,
    blas_int const *n, blas_int const *kd,
    blas_complex_double const *alpha,
    blas_complex_double const *AB, blas_int const *ldab,
    blas_complex_double const *B,  blas_int const *incb,
    blas_complex_double const *beta,
    blas_complex_double       *C,  blas_int const *incc );

}  // extern "C"

// -----------------------------------------------------------------------------
/// @ingroup hbmv
inline void hbmv(
    blas::Uplo uplo,
    int64_t n, int64_t kd,
    float alpha,
    float const *AB, int64_t ldab,
    float const *B, int64_t incb,
    float beta,
    float       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int kd_   = kd;
    blas_int ldab_ = ldab;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_ssbmv( &uplo_, &n_, &kd_,
                &alpha, AB, &ldab_, B, &incb_, &beta, C, &incc_ );
}

/// @ingroup hbmv
inline void hbmv(
    blas::Uplo uplo,
    int64_t n, int64_t kd,
    double alpha,
    double const *AB, int64_t ldab,
    double const *B, int64_t incb,
    double beta,
    double       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int kd_   = kd;
    blas_int ldab_ = ldab;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_dsbmv( &uplo_, &n_, &kd_,
                &alpha, AB, &ldab_, B, &incb_, &beta, C, &incc_ );
}

/// @ingroup hbmv
inline void hbmv(
    blas::Uplo uplo,
    int64_t n, int64_t kd,
    std::complex<float> alpha,
    std::complex<float> const *AB, int64_t ldab,
    std::complex<float> const *B, int64_t incb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int kd_   = kd;
    blas_int ldab_ = ldab;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_chbmv( &uplo_, &n_, &kd_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) AB, &ldab_,
                (blas_complex_float*) B, &incb_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &incc_ );
}

/// @ingroup hbmv
inline void hbmv(
    blas::Uplo uplo,
    int64_t n, int64_t kd,
    std::complex<double> alpha,
    std::complex<double> const *AB, int64_t ldab,
    std::complex<double> const *B, int64_t incb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int kd_   = kd;
    blas_int ldab_ = ldab;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_zhbmv( &uplo_, &n_, &kd_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) AB, &ldab_,
                (blas_complex_double*) B, &incb_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &incc_ );
}

// -----------------------------------------------------------------------------
/// Convenience wrapper loops over hbmv; not intended to be high performance.
/// A is m-by-m, B and C are m-by-n.
/// @ingroup hbmm
template <typename scalar_t>
inline void hbmm(
    blas::Uplo uplo,
    int64_t m, int64_t n, int64_t kd,
    scalar_t alpha,
    scalar_t const *AB, int64_t ldab,
    scalar_t const *B,  int64_t ldb,
    scalar_t beta,
    scalar_t       *C,  int64_t ldc )
{
    for (int64_t j = 0; j < n; ++j)  {
        hbmv( uplo, m, kd,
              alpha, AB, ldab, &B[ j*ldb ], 1, beta, &C[ j*ldc ], 1 );
    }
}


//==============================================================================
// hpmv

extern "C" {

// -----------------------------------------------------------------------------
#define BLAS_sspmv BLAS_FORTRAN_NAME( sspmv, SSpMV )
void BLAS_sspmv(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *AP,
    float const *B,  blas_int const *incb,
    float const *beta,
    float       *C,  blas_int const *incc );

#define BLAS_dspmv BLAS_FORTRAN_NAME( dspmv, DSpMV )
void BLAS_dspmv(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *AP,
    double const *B,  blas_int const *incb,
    double const *beta,
    double       *C,  blas_int const *incc );

#define BLAS_chpmv BLAS_FORTRAN_NAME( chpmv, CHpMV )
void BLAS_chpmv(
    char const *uplo,
    blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *AP,
    blas_complex_float const *B,  blas_int const *incb,
    blas_complex_float const *beta,
    blas_complex_float       *C,  blas_int const *incc );

#define BLAS_zhpmv BLAS_FORTRAN_NAME( zhpmv, ZHpMV )
void BLAS_zhpmv(
    char const *uplo,
    blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *AP,
    blas_complex_double const *B,  blas_int const *incb,
    blas_complex_double const *beta,
    blas_complex_double       *C,  blas_int const *incc );

}  // extern "C"

// -----------------------------------------------------------------------------
/// @ingroup hpmv
inline void hpmv(
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *AP,
    float const *B, int64_t incb,
    float beta,
    float       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_sspmv( &uplo_, &n_,
                &alpha, AP, B, &incb_, &beta, C, &incc_ );
}

/// @ingroup hpmv
inline void hpmv(
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *AP,
    double const *B, int64_t incb,
    double beta,
    double       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_dspmv( &uplo_, &n_,
                &alpha, AP, B, &incb_, &beta, C, &incc_ );
}

/// @ingroup hpmv
inline void hpmv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *AP,
    std::complex<float> const *B, int64_t incb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_chpmv( &uplo_, &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) AP,
                (blas_complex_float*) B, &incb_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &incc_ );
}

/// @ingroup hpmv
inline void hpmv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *AP,
    std::complex<double> const *B, int64_t incb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t incc )
{
    char uplo_     = uplo2char( uplo );
    blas_int n_    = n;
    blas_int incb_ = incb;
    blas_int incc_ = incc;
    BLAS_zhpmv( &uplo_, &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) AP,
                (blas_complex_double*) B, &incb_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &incc_ );
}

// -----------------------------------------------------------------------------
/// Convenience wrapper loops over hpmv; not intended to be high performance.
/// A is m-by-m, B and C are m-by-n.
/// @ingroup hpmm
template <typename scalar_t>
inline void hpmm(
    blas::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const *AP,
    scalar_t const *B, int64_t ldb,
    scalar_t beta,
    scalar_t       *C, int64_t ldc )
{
    for (int64_t j = 0; j < n; ++j)  {
        hpmv( uplo, m,
              alpha, AP, &B[ j*ldb ], 1, beta, &C[ j*ldc ], 1 );
    }
}

}  // namespace blas

#endif // LAPACK_BLAS_WRAPPERS_HH

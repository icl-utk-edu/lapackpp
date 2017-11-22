#ifndef ICL_LAPACK_WRAPPERS_HH
#define ICL_LAPACK_WRAPPERS_HH

#include "lapack_util.hh"

namespace lapack {

// -----------------------------------------------------------------------------
int64_t gesv(
    int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    int64_t* ipiv,
    float* B, int64_t ldb );

int64_t gesv(
    int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    int64_t* ipiv,
    double* B, int64_t ldb );

int64_t gesv(
    int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<float>* B, int64_t ldb );

int64_t gesv(
    int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<double>* B, int64_t ldb );

int64_t gesv(
    int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    int64_t* ipiv,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    int64_t* iter );

int64_t gesv(
    int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    int64_t* iter );

// -----------------------------------------------------------------------------
int64_t getrf(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    int64_t* ipiv );

int64_t getrf(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    int64_t* ipiv );

int64_t getrf(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv );

int64_t getrf(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv );

// -----------------------------------------------------------------------------
int64_t getri(
    int64_t n,
    float* A, int64_t lda,
    int64_t const* ipiv );

int64_t getri(
    int64_t n,
    double* A, int64_t lda,
    int64_t const* ipiv );

int64_t getri(
    int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t const* ipiv );

int64_t getri(
    int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t const* ipiv );

// -----------------------------------------------------------------------------
int64_t getrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    float const* A, int64_t lda,
    int64_t const* ipiv,
    float* B, int64_t ldb );

int64_t getrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    double const* A, int64_t lda,
    int64_t const* ipiv,
    double* B, int64_t ldb );

int64_t getrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    int64_t const* ipiv,
    std::complex<float>* B, int64_t ldb );

int64_t getrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    int64_t const* ipiv,
    std::complex<double>* B, int64_t ldb );

// -----------------------------------------------------------------------------
int64_t gerfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    float const* A, int64_t lda,
    float const* AF, int64_t ldaf,
    int64_t const* ipiv,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* ferr,
    float* berr );

int64_t gerfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    double const* A, int64_t lda,
    double const* AF, int64_t ldaf,
    int64_t const* ipiv,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* ferr,
    double* berr );

int64_t gerfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* AF, int64_t ldaf,
    int64_t const* ipiv,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* ferr,
    float* berr );

int64_t gerfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* AF, int64_t ldaf,
    int64_t const* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* ferr,
    double* berr );

// -----------------------------------------------------------------------------
int64_t gecon(
    lapack::Norm norm, int64_t n,
    float const* A, int64_t lda, float anorm,
    float* rcond );

int64_t gecon(
    lapack::Norm norm, int64_t n,
    double const* A, int64_t lda, double anorm,
    double* rcond );

int64_t gecon(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* A, int64_t lda, float anorm,
    float* rcond );

int64_t gecon(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* A, int64_t lda, double anorm,
    double* rcond );

// -----------------------------------------------------------------------------
int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb );

int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb );

int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb );

int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb );

int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    int64_t* iter );

int64_t posv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    int64_t* iter );

// -----------------------------------------------------------------------------
int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda );

int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda );

int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda );

int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda );

// -----------------------------------------------------------------------------
int64_t potrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float const* A, int64_t lda,
    float* B, int64_t ldb );

int64_t potrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double const* A, int64_t lda,
    double* B, int64_t ldb );

int64_t potrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>* B, int64_t ldb );

int64_t potrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>* B, int64_t ldb );

// -----------------------------------------------------------------------------
int64_t potri(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda );

int64_t potri(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda );

int64_t potri(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda );

int64_t potri(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda );

// -----------------------------------------------------------------------------
int64_t porfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float const* A, int64_t lda,
    float const* AF, int64_t ldaf,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* ferr,
    float* berr );

int64_t porfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double const* A, int64_t lda,
    double const* AF, int64_t ldaf,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* ferr,
    double* berr );

int64_t porfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* AF, int64_t ldaf,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* ferr,
    float* berr );

int64_t porfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* AF, int64_t ldaf,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* ferr,
    double* berr );

// -----------------------------------------------------------------------------
int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb );

int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb );

int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb );

int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb );

// -----------------------------------------------------------------------------
int64_t geqrf(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* tau );

int64_t geqrf(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* tau );

int64_t geqrf(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* tau );

int64_t geqrf(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* tau );

// -----------------------------------------------------------------------------
int64_t orgqr(
    int64_t m, int64_t n, int64_t k,
    float* A, int64_t lda,
    float const* tau );

int64_t orgqr(
    int64_t m, int64_t n, int64_t k,
    double* A, int64_t lda,
    double const* tau );

// -----------------------------------------------------------------------------
int64_t ormqr(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    float const* A, int64_t lda,
    float const* tau,
    float* C, int64_t ldc );

int64_t ormqr(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    double const* A, int64_t lda,
    double const* tau,
    double* C, int64_t ldc );

// -----------------------------------------------------------------------------
float lange(
    lapack::Norm norm, int64_t m, int64_t n,
    float const* A, int64_t lda );

double lange(
    lapack::Norm norm, int64_t m, int64_t n,
    double const* A, int64_t lda );

float lange(
    lapack::Norm norm, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda );

double lange(
    lapack::Norm norm, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda );

// -----------------------------------------------------------------------------
float lanhe(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda );

double lanhe(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda );

// -----------------------------------------------------------------------------
float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda );

double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda );

float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda );

double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda );

// -----------------------------------------------------------------------------
void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    float* X );

void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    double* X );

void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    std::complex<float>* X );

void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    std::complex<double>* X );

}  // namespace lapack

#endif  // ICL_LAPACK_WRAPPERS_HH

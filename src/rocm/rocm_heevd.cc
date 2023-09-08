
#include "lapack/defines.h"

#if defined(LAPACK_HAVE_ROCBLAS)

#include "rocm_common.hh"

namespace blas {
namespace internal {

rocblas_fill uplo2rocblas(blas::Uplo uplo);

}
}


//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Specified scalar_t float -> syevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    *dev_work_size = n * sizeof(float);
    *host_work_size = 0;
}

//----------
// Specified scalar_t double -> dyevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    *dev_work_size = n * sizeof(double);
    *host_work_size = 0;
}

//----------
// Specified scalar_t std::complex<float> -> heevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    *dev_work_size = n * sizeof(std::complex<float>);
    *host_work_size = 0;
}

//----------
// Specified scalar_t std::complex<double> -> heevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    *dev_work_size = n * sizeof(std::complex<double>);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Templated for scalar_t
template <typename scalar_t>
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // call scalar_t specified routines
    heevd_work_size_bytes(
        jobz, uplo, n, dA, ldda, dW, dev_work_size, host_work_size, queue );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around rocSolver to deal with precisions.
rocblas_status  rocsolver_heevd(
    rocblas_handle solver, const rocblas_evect jobz,
    const rocblas_fill uplo, rocblas_int n, float* dA,
    const rocblas_int ldda, float* dW, float* dev_work, rocblas_int* info )
{
    return rocsolver_ssyevd(
        solver, jobz, uplo, n, dA, ldda, dW, dev_work, info );
}

//----------
rocblas_status  rocsolver_heevd(
    rocblas_handle solver, const rocblas_evect jobz,
    const rocblas_fill uplo, rocblas_int n, double* dA,
    const rocblas_int ldda, double* dW, double* dev_work, rocblas_int* info )
{
    return rocsolver_dsyevd(
        solver, jobz, uplo, n, dA, ldda, dW, dev_work, info );
}

//----------
rocblas_status  rocsolver_heevd(
    rocblas_handle solver, const rocblas_evect jobz,
    const rocblas_fill uplo, rocblas_int n, std::complex<float>* dA,
    const rocblas_int ldda, float* dW, float* dev_work, rocblas_int* info )
{
    return rocsolver_cheevd(
        solver, jobz, uplo, n, (rocblas_float_complex*) dA, ldda,
        dW, dev_work, info );
}

//----------
rocblas_status  rocsolver_heevd(
    rocblas_handle solver, const rocblas_evect jobz,
    const rocblas_fill uplo, rocblas_int n, std::complex<double>* dA,
    const rocblas_int ldda, double* dW, double* dev_work, rocblas_int* info )
{
    return rocsolver_zheevd(
        solver, jobz, uplo, n, (rocblas_double_complex*) dA, ldda,
        dW, dev_work, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    using real_t = blas::real_type<scalar_t>;
    auto solver = queue.handle();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    //int lwork = dev_work_size / sizeof(scalar_t);
    blas_dev_call(
        rocsolver_heevd(
            solver, job2eigmode_rocsolver(jobz), blas::internal::uplo2rocblas(uplo), n, dA, ldda, dW,
            (real_t*) dev_work, dev_info ));
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, float* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, double* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, std::complex<float>* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, std::complex<double>* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

//--------------------
template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_CUBLAS


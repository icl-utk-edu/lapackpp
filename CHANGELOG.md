2023.08.25
  - Use yyyy.mm.dd version scheme, instead of yyyy.mm.release
  - Added oneAPI support to CMake
  - Fixed int64 support
  - More robust Makefile configure doesn't require CUDA or ROCm to be in
    compiler search paths (CPATH, LIBRARY_PATH, etc.)
  - Added `gemqrt` to multiply by Q from QR

2023.06.00
  - Updates for BLAS++ changes to Queue class

2023.01.00
  - Added oneAPI port (currently Makefile only)
  - Added `{or,un}hr_col` Householder reconstruction
  - Added `tgexc, tgsen` to reorder generalized Schur form
  - Added `lartg` to generate plane rotation
  - Moved main repo to https://github.com/icl-utk-edu/lapackpp/
  - Use python3

2022.07.00
  - Added device queue and Cholesky (potrf), LU (getrf), and QR (geqrf) on GPU
    for CUDA (cuSolver) and ROCm (rocSolver)
  - Added geqr tester

2022.05.00
  - Added laed4, sturm
  - Use custom allocator to avoid workspace initialization overhead
  - Backward error checks for more routines

2021.04.00
  - Added include/lapack/defines.h based on configuration
  - Added larfgp
  - More robust backward error checks
  - Makefile and CMake fixes

2020.10.01
  - Fixes: ILP64, CMake output padding

2020.10.00
  - Fixes: CMake version
  - Added `make check`

2020.09.00
  - Initial release
    - Supports LAPACK >= 3.2.1
    - Includes routines through LAPACK 3.7.0
    - Makefile and CMake build options

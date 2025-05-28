2025.05.28 (ABI 2.0.0)
  - Added support for BLIS and libFLAME (hence AOCL)
  - Removed support for ACML
  - Removed [cz]symv and [cz]syr; moved them to BLAS++
  - Removed deprecated `<enum>2str`, `str2<enum>`, `char2<enum>`, `<enum>2char`
  - Tester prints stats with --repeat
  - Fixed SYCL include path
  - Fixed conflict between LAPACKE and LAPACK++ headers

2024.10.26 (ABI 1.0.0)
  - Added eigenvalue utilities (lae2, laev2, lasr)
  - Refactor eigenvalue testers
  - Use std::hypot instead of lapy2, lapy3. Deprecate lapy2, lapy3
  - Use to_lapack_int to convert int32 to int64

2024.05.31 (ABI 1.0.0)
  - Added shared library ABI version
  - Updated enum parameters to have `to_string`, `from_string`;
    deprecate `<enum>2str`, `str2<enum>`, `char2<enum>`, `<enum>2char`
  - Removed some deprecated functions

2023.11.05
  - Add heevd GPU wrapper for CUDA, ROCm, oneMKL
  - Update Fortran strlen handling
  - Fix CMake library ordering

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

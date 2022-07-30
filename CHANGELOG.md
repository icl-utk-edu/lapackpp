2022.07.00
  - Added device queue and Cholesky (potrf), LU (getrf), and QR (geqrf) on GPU
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
  - Initial release.
    - Supports LAPACK >= 3.2.1.
    - Includes routines through LAPACK 3.7.0.
    - Makefile and CMake build options

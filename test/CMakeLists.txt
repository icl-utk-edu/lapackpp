# CMake script for LAPACK++ testers
# Requires Libtest, BLASPP, and LAPACKPP

cmake_minimum_required(VERSION 3.2)

add_executable(lapackpp_test
    matrix_generator.cc
    matrix_params.cc
    test.cc
    test_gbcon.cc
    test_gbequ.cc
    test_gbrfs.cc
    test_gbsv.cc
    test_gbtrf.cc
    test_gbtrs.cc
    test_gecon.cc
    test_geequ.cc
    test_geev.cc
    test_gehrd.cc
    test_gelqf.cc
    test_gels.cc
    test_gelsd.cc
    test_gelss.cc
    test_gelsy.cc
    test_geqlf.cc
    test_geqrf.cc
    test_gerfs.cc
    test_gerqf.cc
    test_gesdd.cc
    test_gesv.cc
    test_gesvd.cc
    test_gesvdx.cc
    test_gesvx.cc
    test_getrf.cc
    test_getri.cc
    test_getrs.cc
    test_getsls.cc
    test_ggev.cc
    test_ggglm.cc
    test_gglse.cc
    test_ggqrf.cc
    test_ggrqf.cc
    test_gtcon.cc
    test_gtrfs.cc
    test_gtsv.cc
    test_gttrf.cc
    test_gttrs.cc
    test_hbev.cc
    test_hbevd.cc
    test_hbevx.cc
    test_hbgv.cc
    test_hbgvd.cc
    test_hbgvx.cc
    test_hecon.cc
    test_heev.cc
    test_heevd.cc
    test_heevr.cc
    test_heevx.cc
    test_hegst.cc
    test_hegv.cc
    test_hegvd.cc
    test_hegvx.cc
    test_herfs.cc
    test_hesv.cc
    test_hetrd.cc
    test_hetrf.cc
    test_hetri.cc
    test_hetrs.cc
    test_hpcon.cc
    test_hpev.cc
    test_hpevd.cc
    test_hpevx.cc
    test_hpgst.cc
    test_hpgv.cc
    test_hpgvd.cc
    test_hpgvx.cc
    test_hprfs.cc
    test_hpsv.cc
    test_hptrd.cc
    test_hptrf.cc
    test_hptri.cc
    test_hptrs.cc
    test_lacpy.cc
    test_langb.cc
    test_lange.cc
    test_langt.cc
    test_lanhb.cc
    test_lanhe.cc
    test_lanhp.cc
    test_lanhs.cc
    test_lanst.cc
    test_lanht.cc
    test_lansb.cc
    test_lansp.cc
    test_lansy.cc
    test_lantb.cc
    test_lantp.cc
    test_lantr.cc
    test_larf.cc
    test_larfb.cc
    test_larfg.cc
    test_larft.cc
    test_larfx.cc
    test_larfy.cc
    test_laset.cc
    test_laswp.cc
    test_pbcon.cc
    test_pbequ.cc
    test_pbrfs.cc
    test_pbsv.cc
    test_pbtrf.cc
    test_pbtrs.cc
    test_pocon.cc
    test_poequ.cc
    test_porfs.cc
    test_posv.cc
    test_potrf.cc
    test_potri.cc
    test_potrs.cc
    test_ppcon.cc
    test_ppequ.cc
    test_pprfs.cc
    test_ppsv.cc
    test_pptrf.cc
    test_pptri.cc
    test_pptrs.cc
    test_ptcon.cc
    test_ptrfs.cc
    test_ptsv.cc
    test_pttrf.cc
    test_pttrs.cc
    test_spcon.cc
    test_sprfs.cc
    test_spsv.cc
    test_sptrf.cc
    test_sptri.cc
    test_sptrs.cc
    test_sycon.cc
    test_syr.cc
    test_syrfs.cc
    test_sysv.cc
    test_sysv_aa.cc
    test_sysv_rk.cc
    test_sysv_rook.cc
    test_sytrf.cc
    test_sytrf_aa.cc
    test_sytrf_rk.cc
    test_sytrf_rook.cc
    test_sytri.cc
    test_sytrs.cc
    test_sytrs_aa.cc
    test_sytrs_rook.cc
    test_unghr.cc
    test_unglq.cc
    test_ungql.cc
    test_ungqr.cc
    test_ungrq.cc
    test_ungtr.cc
    test_unmhr.cc
    test_unmtr.cc
    test_upgtr.cc
    test_upmtr.cc
    # Triangle-pentagon QR/LQ
    test_tplqt.cc
    test_tplqt2.cc
    test_tpmlqt.cc
    test_tpmqrt.cc
    test_tpqrt.cc
    test_tpqrt2.cc
    test_tprfb.cc
    test_symv.cc
)

# Check if user supplied LIBTEST_DIR on the command line
# Set -Dlibtest_DIR=/path/to/libtest
if(libtest_DIR)
    message("Found libtest_DIR")
else()
    find_path(libtest_DIR libtestConfig.cmake
        PATHS
        ${CMAKE_INSTALL_PREFIX}
        ${CMAKE_INSTALL_PREFIX}/test
        )
    message("libtest_DIR: " ${libtest_DIR})
endif()

# Check if user supplied BLASPP_DIR on the command line
if(BLASPP_DIR)
    message("using user supplied BLASPP_DIR: ${BLASPP_DIR}")
    set(blaspp_DIR "${BLASPP_DIR}")
endif()

find_package(libtest)
find_package(blaspp)
#find_package(LAPACKE REQUIRED)

cmake_policy(SET CMP0079 NEW)

if ("${LAPACKE_DEFINES}" MATCHES "HAVE_LAPACKE")
    #message (" yes lapacke")
    #message ("lapacke include dirs: ${LAPACKE_INCLUDE_DIRS}")
    target_link_libraries(lapackpp_test
        PUBLIC
            ${LAPACKE_LIBRARIES}
    )
    target_include_directories(lapackpp_test
        PUBLIC ${LAPACKE_INCLUDE_DIRS}
        )
endif()

target_link_libraries(lapackpp_test
    PUBLIC
    libtest
    gfortran
    #blaspp
    lapackpp
    #LAPACKE
)

target_include_directories(lapackpp_test
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}../include
    ${LAPACKE_INCLUDE_DIRS}
)

# Some debug
if(LAPACKPP_TEST_DEBUG)
    message("LAPACKPP tester compilation definitions:")
    message("MKL INTS defines: ${BLAS_int}")
    message("FORTRAN MANGLING defines: " ${FORTRAN_MANGLING_DEFINES})
    message("BLAS DEFINES: " ${BLAS_DEFINES})
    message("CBLAS DEFINES: " ${CBLAS_DEFINES})
    message("BLAS RETURN: " ${BLAS_RETURN})
    message("LAPACK DEFINES: " ${LAPACK_DEFINES})
    message("LAPACKE DEFINES: " ${LAPACKE_DEFINES})
    message("XBLAS DEFINES: " ${XBLAS_DEFINES})
    message("LAPACK version define: " ${LAPACK_VER_DEFINE})
endif()

# these definitions are inherited from LAPACKPP
#target_compile_definitions(lapackpp_test PRIVATE
    # Example definition set:
    # -DFORTRAN_ADD_ -DADD_ -DHAVE_BLAS -DBLAS_COMPLEX_RETURN_ARGUMENT -DHAVE_MKL -DHAVE_CBLAS -DHAVE_LAPACK -DLAPACK_VERSION=30700 -DHAVE_XBLAS -DHAVE_LAPACKE
#    ${FORTRAN_MANGLING_DEFINES}
#    ${BLAS_DEFINES}
#    ${BLAS_RETURN}
#    ${MKL_DEFINES}
#    ${CBLAS_DEFINES}
#    ${LAPACK_DEFINES}
#    ${BLAS_INT_DEFINES}
#    ${LAPACK_VER_DEFINE}
#    ${XBLAS_DEFINES}
#    ${LAPACKE_DEFINES}
#)

# Move test script to build directory
add_custom_command(TARGET lapackpp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

#Move test script to build directory
add_custom_command(TARGET lapackpp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/lapackpp_test
           ${CMAKE_CURRENT_BINARY_DIR}/test
)

if(LAPACKPP_TEST_DEBUG)
include(CMakePrintHelpers)
cmake_print_properties(TARGETS lapackpp_test PROPERTIES
    INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
endif()

# Since we can't do 'make test'...
add_custom_target(testing DEPENDS lapackpp_test)

# Move test script to build directory
add_custom_command(TARGET lapackpp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

#Move test script to build directory
add_custom_command(TARGET lapackpp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/lapackpp_test
           ${CMAKE_CURRENT_BINARY_DIR}/test
)

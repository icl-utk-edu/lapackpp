# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Can't build testers if CBLAS, LAPACKE, or TestSweeper are not found.
if (NOT blaspp_cblas_found)
    message( WARNING "CBLAS not found; tester cannot be built." )
    return()
endif()

if (NOT lapacke_found)
    message( WARNING "LAPACKE not found; tester cannot be built." )
    return()
endif()

# Search for TestSweeper library, if not already included (e.g., in SLATE).
message( STATUS "Checking for TestSweeper library" )
if (NOT TARGET testsweeper)
    find_package( testsweeper QUIET )
    if (testsweeper_FOUND)
        message( "   Found TestSweeper library: ${testsweeper_DIR}" )
    else()
        set( url "https://github.com/icl-utk-edu/testsweeper" )
        set( tag "v2023.08.25" )
        message( "" )
        message( "---------- TestSweeper" )
        message( STATUS "Fetching TestSweeper ${tag} from ${url}" )
        include( FetchContent )
        FetchContent_Declare( testsweeper GIT_REPOSITORY "${url}"
                                          GIT_TAG "${tag}" )
        FetchContent_MakeAvailable( testsweeper )
        message( "---------- TestSweeper done" )
        message( "" )
    endif()
else()
    message( "   TestSweeper already included" )
endif()

#-------------------------------------------------------------------------------
set( tester "${lapackpp_}tester" )
add_executable(
    ${tester}
    cblas_wrappers.cc
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
    test_gemqrt.cc
    test_geqlf.cc
    test_geqr.cc
    test_geqrf.cc
    test_geqrf_device.cc
    test_gerfs.cc
    test_gerqf.cc
    test_gesdd.cc
    test_gesv.cc
    test_gesvd.cc
    test_gesvdx.cc
    test_gesvx.cc
    test_getrf.cc
    test_getrf_device.cc
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
    test_laed4.cc
    test_langb.cc
    test_lange.cc
    test_langt.cc
    test_lanhb.cc
    test_lanhe.cc
    test_lanhp.cc
    test_lanhs.cc
    test_lanht.cc
    test_lansb.cc
    test_lansp.cc
    test_lanst.cc
    test_lansy.cc
    test_lantb.cc
    test_lantp.cc
    test_lantr.cc
    test_larf.cc
    test_larfb.cc
    test_larfg.cc
    test_larfgp.cc
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
    test_potrf_device.cc
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
    test_sturm.cc
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
    test_tgexc.cc
    test_tgsen.cc
    test_unghr.cc
    test_unglq.cc
    test_ungql.cc
    test_ungqr.cc
    test_ungrq.cc
    test_ungtr.cc
    test_unhr_col.cc    test_orhr_col.cc
    test_unmhr.cc
    test_unmtr.cc
    test_upgtr.cc
    test_upmtr.cc
    test_tplqt.cc
    test_tplqt2.cc
    test_tpmlqt.cc
    test_tpmqrt.cc
    test_tpqrt.cc
    test_tpqrt2.cc
    test_tprfb.cc
    test_symv.cc
    test_larfy.cc
)

# C++11 is inherited from blaspp, but disabling extensions is not.
set_target_properties( ${tester} PROPERTIES CXX_EXTENSIONS false )

target_link_libraries(
    ${tester}
    testsweeper
    lapackpp
    ${blaspp_cblas_libraries}
    ${lapacke_libraries}
)

target_include_directories(
    ${tester}
    PRIVATE
        "${blaspp_cblas_include}"
        "${lapacke_include}"
)

if (gpu_backend STREQUAL "sycl" )
    # Avoid "comparison with NaN" warnings from the IntelLLVM compiler
    # while compiling test/matrix_generator.cc (the compiler uses fast
    # floating point mode by default).
   target_compile_options( ${tester} PRIVATE -fp-model=precise )
endif()

# Copy run_tests script to build directory.
add_custom_command(
    TARGET ${tester} POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

if (lapackpp_is_project)
    # 'make check' tests subset of routines, to avoid spurious failures
    add_custom_target(
        "check"
        COMMAND
            python3 run_tests.py --quick
                gesv getrf posv potrf geqrf ungqr gels
                geev heev heevd heevr gesvd
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

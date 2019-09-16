# CMake script for LAPACK++ library
# repo: http://bitbucket.org/icl/lapackpp
# Requires Libtest (for building tests) and BLASPP library
#   - http://bitbucket.org/icl/libtest
#   - http://bitbucket.org/icl/blaspp

# Build and install:
#   hg clone http://bitbucket.org/icl/lapackpp
#   mkdir build
#   cd build
#   cmake -Dlibtest_DIR=/path/to/libtest -Dblaspp_DIR=/path/to/blaspp ..
#   make install

cmake_minimum_required(VERSION 3.2)

project(LAPACKPP CXX)

# BLAS options
# todo: Goto, BLIS, FLAME, others?
set(BLAS_LIBRARY "auto" CACHE STRING
    "Choose BLAS & LAPACK library: AMD, Apple, Cray, IBM, Intel, OpenBLAS" )
set_property(CACHE BLAS_LIBRARY PROPERTY STRINGS
    "auto" "AMD ACML" "Apple Accelerate" "Cray LibSci" "IBM ESSL"
    "Intel MKL" "OpenBLAS" )

set(BLAS_LIBRARY_MKL "auto" CACHE STRING
    "For Intel MKL: use Intel ifort or GNU gfortran conventions?" )
set_property(CACHE BLAS_LIBRARY_MKL PROPERTY STRINGS
    "auto" "GNU gfortran conventions" "Intel ifort conventions" )

set(BLAS_LIBRARY_INTEGER "auto" CACHE STRING
    "BLAS integer size: int (LP64) or int64_t (ILP64)" )
set_property(CACHE BLAS_LIBRARY_INTEGER PROPERTY STRINGS
    "auto" "int (LP64)" "int64_t (ILP64)" )

set(BLAS_LIBRARY_THREADING "auto" CACHE STRING
    "BLAS threaded or sequential" )
set_property(CACHE BLAS_LIBRARY_THREADING PROPERTY STRINGS
    "auto" "threaded" "sequential" )

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")

# Build the testers by default
option(BUILD_LAPACKPP_TESTS "Build LAPACK++ tests - has more reqs" ON)

# Enforce out of source builds
string (TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" SOURCE_DIR_LOWER)
string (TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" BINARY_DIR_LOWER )
if (SOURCE_DIR_LOWER STREQUAL BINARY_DIR_LOWER)
    message (FATAL_ERROR "Compiling LAPACKPP with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make")
endif ()

if (CMAKE_HOST_APPLE)
    set (CMAKE_MACOSX_RPATH 1)
endif()

add_library(lapackpp
    src/bbcsd.cc
    src/bdsdc.cc
    src/bdsqr.cc
    src/bdsvdx.cc
    src/disna.cc
    src/gbbrd.cc
    src/gbcon.cc
    src/gbequ.cc
    src/gbequb.cc
    src/gbrfs.cc
    src/gbrfsx.cc
    src/gbsv.cc
    src/gbsvx.cc
    src/gbtrf.cc
    src/gbtrs.cc
    src/gebak.cc
    src/gebal.cc
    src/gebrd.cc
    src/gecon.cc
    src/geequ.cc
    src/geequb.cc
    src/gees.cc
    src/geesx.cc
    src/geev.cc
    src/gehrd.cc
    src/gelq.cc
    src/gelq2.cc
    src/gelqf.cc
    src/gels.cc
    src/gelsd.cc
    src/gelss.cc
    src/gelsy.cc
    src/gemlq.cc
    src/gemqr.cc
    src/geql2.cc
    src/geqlf.cc
    src/geqp3.cc
    src/geqr.cc
    src/geqr2.cc
    src/geqrf.cc
    src/geqrfp.cc
    src/geqrt.cc
    src/geqrt2.cc
    src/geqrt3.cc
    src/gerfs.cc
    src/gerfsx.cc
    src/gerq2.cc
    src/gerqf.cc
    src/gesdd.cc
    src/gesv.cc
    src/gesvd.cc
    src/gesvdx.cc
    src/gesvx.cc
    src/getf2.cc
    src/getrf.cc
    src/getrf2.cc
    src/getri.cc
    src/getrs.cc
    src/getsls.cc
    src/ggbak.cc
    src/ggbal.cc
    src/gges.cc
    src/gges3.cc
    src/ggesx.cc
    src/ggev.cc
    src/ggev3.cc
    src/ggglm.cc
    src/gghrd.cc
    src/gglse.cc
    src/ggqrf.cc
    src/ggrqf.cc
    src/ggsvd3.cc
    src/ggsvp3.cc
    src/gtcon.cc
    src/gtrfs.cc
    src/gtsv.cc
    src/gtsvx.cc
    src/gttrf.cc
    src/gttrs.cc
    src/hbev.cc
    src/hbev_2stage.cc
    src/hbevd.cc
    src/hbevd_2stage.cc
    src/hbevx.cc
    src/hbevx_2stage.cc
    src/hbgst.cc
    src/hbgv.cc
    src/hbgvd.cc
    src/hbgvx.cc
    src/hbtrd.cc
    src/hecon.cc
    src/hecon_rk.cc
    src/heequb.cc
    src/heev.cc
    src/heev_2stage.cc
    src/heevd.cc
    src/heevd_2stage.cc
    src/heevr.cc
    src/heevr_2stage.cc
    src/heevx.cc
    src/heevx_2stage.cc
    src/hegst.cc
    src/hegv.cc
    src/hegv_2stage.cc
    src/hegvd.cc
    src/hegvx.cc
    src/herfs.cc
    src/herfsx.cc
    src/hesv.cc
    src/hesv_aa.cc
    src/hesv_rk.cc
    src/hesv_rook.cc
    src/hesvx.cc
    src/heswapr.cc
    src/hetrd.cc
    src/hetrd_2stage.cc
    src/hetrf.cc
    src/hetrf_aa.cc
    src/hetrf_rk.cc
    src/hetrf_rook.cc
    src/hetri.cc
    src/hetri2.cc
    src/hetri_rk.cc
    src/hetrs.cc
    src/hetrs2.cc
    src/hetrs_aa.cc
    src/hetrs_rk.cc
    src/hetrs_rook.cc
    src/hfrk.cc
    src/hgeqz.cc
    src/hpcon.cc
    src/hpev.cc
    src/hpevd.cc
    src/hpevx.cc
    src/hpgst.cc
    src/hpgv.cc
    src/hpgvd.cc
    src/hpgvx.cc
    src/hprfs.cc
    src/hpsv.cc
    src/hpsvx.cc
    src/hptrd.cc
    src/hptrf.cc
    src/hptri.cc
    src/hptrs.cc
    src/hseqr.cc
    src/lacgv.cc
    src/lacp2.cc
    src/lacpy.cc
    src/lag2c.cc
    src/lag2d.cc
    src/lag2s.cc
    src/lag2z.cc
    src/lagge.cc
    src/laghe.cc
    src/lagsy.cc
    src/lamch.cc
    src/langb.cc
    src/lange.cc
    src/langt.cc
    src/lanhb.cc
    src/lanhe.cc
    src/lanhp.cc
    src/lanhs.cc
    src/lanht.cc
    src/lansb.cc
    src/lansp.cc
    src/lanst.cc
    src/lansy.cc
    src/lantb.cc
    src/lantp.cc
    src/lantr.cc
    src/lapmr.cc
    src/lapmt.cc
    src/lapy2.cc
    src/lapy3.cc
    src/larf.cc
    src/larfb.cc
    src/larfg.cc
    src/larft.cc
    src/larfx.cc
    src/larnv.cc
    src/lartgp.cc
    src/lartgs.cc
    src/lascl.cc
    src/laset.cc
    src/lassq.cc
    src/laswp.cc
    src/lauum.cc
    src/opgtr.cc
    src/opmtr.cc
    src/orcsd2by1.cc
    src/orgbr.cc
    src/orghr.cc
    src/orglq.cc
    src/orgql.cc
    src/orgqr.cc
    src/orgrq.cc
    src/orgtr.cc
    src/ormbr.cc
    src/ormhr.cc
    src/ormlq.cc
    src/ormql.cc
    src/ormqr.cc
    src/ormrq.cc
    src/ormrz.cc
    src/ormtr.cc
    src/pbcon.cc
    src/pbequ.cc
    src/pbrfs.cc
    src/pbstf.cc
    src/pbsv.cc
    src/pbsvx.cc
    src/pbtrf.cc
    src/pbtrs.cc
    src/pftrf.cc
    src/pftri.cc
    src/pftrs.cc
    src/pocon.cc
    src/poequ.cc
    src/poequb.cc
    src/porfs.cc
    src/porfsx.cc
    src/posv.cc
    src/posvx.cc
    src/potf2.cc
    src/potrf.cc
    src/potrf2.cc
    src/potri.cc
    src/potrs.cc
    src/ppcon.cc
    src/ppequ.cc
    src/pprfs.cc
    src/ppsv.cc
    src/ppsvx.cc
    src/pptrf.cc
    src/pptri.cc
    src/pptrs.cc
    src/pstrf.cc
    src/ptcon.cc
    src/pteqr.cc
    src/ptrfs.cc
    src/ptsv.cc
    src/ptsvx.cc
    src/pttrf.cc
    src/pttrs.cc
    src/sbev.cc
    src/sbev_2stage.cc
    src/sbevd.cc
    src/sbevd_2stage.cc
    src/sbevx.cc
    src/sbevx_2stage.cc
    src/sbgst.cc
    src/sbgv.cc
    src/sbgvd.cc
    src/sbgvx.cc
    src/sbtrd.cc
    src/sfrk.cc
    src/spcon.cc
    src/spev.cc
    src/spevd.cc
    src/spevx.cc
    src/spgst.cc
    src/spgv.cc
    src/spgvd.cc
    src/spgvx.cc
    src/sprfs.cc
    src/spsv.cc
    src/spsvx.cc
    src/sptrd.cc
    src/sptrf.cc
    src/sptri.cc
    src/sptrs.cc
    src/stedc.cc
    src/stegr.cc
    src/stein.cc
    src/stemr.cc
    src/steqr.cc
    src/sterf.cc
    src/stev.cc
    src/stevd.cc
    src/stevr.cc
    src/stevx.cc
    src/sycon.cc
    src/sycon_rk.cc
    src/syequb.cc
    src/syev.cc
    src/syev_2stage.cc
    src/syevd.cc
    src/syevd_2stage.cc
    src/syevr.cc
    src/syevr_2stage.cc
    src/syevx.cc
    src/syevx_2stage.cc
    src/sygst.cc
    src/sygv.cc
    src/sygv_2stage.cc
    src/sygvd.cc
    src/sygvx.cc
    src/syr.cc
    src/syrfs.cc
    src/syrfsx.cc
    src/sysv.cc
    src/sysv_aa.cc
    src/sysv_rk.cc
    src/sysv_rook.cc
    src/sysvx.cc
    src/syswapr.cc
    src/sytrd.cc
    src/sytrd_2stage.cc
    src/sytrf.cc
    src/sytrf_aa.cc
    src/sytrf_rk.cc
    src/sytrf_rook.cc
    src/sytri.cc
    src/sytri2.cc
    src/sytri_rk.cc
    src/sytrs.cc
    src/sytrs2.cc
    src/sytrs_aa.cc
    src/sytrs_rk.cc
    src/sytrs_rook.cc
    src/tbcon.cc
    src/tbrfs.cc
    src/tbtrs.cc
    src/tfsm.cc
    src/tftri.cc
    src/tfttp.cc
    src/tfttr.cc
    src/tgsja.cc
    src/tgsyl.cc
    src/tpcon.cc
    src/tpqrt.cc
    src/tpqrt2.cc
    src/tprfs.cc
    src/tptri.cc
    src/tptrs.cc
    src/tpttf.cc
    src/tpttr.cc
    src/trcon.cc
    src/trevc.cc
    src/trevc3.cc
    src/trexc.cc
    src/trrfs.cc
    src/trsen.cc
    src/trtri.cc
    src/trtrs.cc
    src/trttf.cc
    src/trttp.cc
    src/tzrzf.cc
    src/ungbr.cc
    src/unghr.cc
    src/unglq.cc
    src/ungql.cc
    src/ungqr.cc
    src/ungrq.cc
    src/ungtr.cc
    src/unmbr.cc
    src/unmhr.cc
    src/unmlq.cc
    src/unmql.cc
    src/unmqr.cc
    src/unmrq.cc
    src/unmrz.cc
    src/unmtr.cc
    src/upgtr.cc
    src/upmtr.cc
    #Triangle-pentagon QR/LQ
    src/tplqt.cc
    src/tplqt2.cc
    src/tpmlqt.cc
    src/tpmqrt.cc
    src/tprfb.cc
    # Moved from BLAS++
    src/symv.cc
    src/larfy.cc
)

# Check user supplied BLAS libraries
if (0)
if(BLAS_LIBRARIES)
    message("BLAS_LIBRARIES FOUND: " ${BLAS_LIBRARIES})
    include(CheckFortranFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(BLAS_FOUND)
        message(STATUS "--> BLAS supplied by user is WORKING, will use ${BLAS_LIBRARIES}.")
        target_link_libraries(blaspp ${BLAS_LIBRARIES})
    else()
        message(ERROR "--> BLAS supplied by user is not WORKING, CANNOT USE ${BLAS_LIBRARIES}.")
        message(ERROR "-->     Will use REFERENCE BLAS (by default)")
        message(ERROR "-->     Or Correct your BLAS_LIBRARIES entry ")
        message(ERROR "-->     Or Consider checking USE_OPTIMIZED_BLAS")
    endif()
    # Search for system installed BLAS
elseif(USE_OPTIMIZED_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message("BLAS_FOUND: " ${BLAS_LIBRARIES})
        target_link_libraries(blaspp ${BLAS_LIBRARIES})
    endif()
else()
    message("BLAS_LIBRARIES not found, using custom search: " ${BLAS_LIBRARIES})

    # Custom BLAS library locator, if all else fails
    include ("cmake/BLASFinder.cmake")
endif()

endif() #check blas libs

# Set -Dblaspp_DIR=/path/to/blaspp or -DBLASPP_DIR=/path/to/blaspp
if(BLASPP_DIR)
    message("using user supplied BLASPP_DIR: ${BLASPP_DIR}")
    #set(blaspp_DIR "${BLASPP_DIR}")
else()
    find_path(blaspp_DIR blasppConfig.cmake
        PATHS
        ${CMAKE_INSTALL_PREFIX}/blaspp
        ${CMAKE_INSTALL_PREFIX}
        )
    #message("blaspp_DIR: " ${blaspp_DIR})
endif()

find_package(blaspp)

if(LAPACKPP_QUIET)
include(CMakePrintHelpers)
cmake_print_properties(TARGETS blaspp PROPERTIES
    INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
cmake_print_properties(TARGETS lapackpp PROPERTIES
    INTERFACE_INCLUDE_DIRS INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
endif()

target_link_libraries(lapackpp PUBLIC blaspp)

get_target_property(blaspp_links blaspp INTERFACE_LINK_LIBRARIES)
get_target_property(blaspp_compile_definitions blaspp COMPILE_DEFINITIONS)
#get_target_property(blaspp_interface_compile_definitions blaspp INTERFACE_COMPILE_DEFINITIONS)

if ("${blaspp_links}" MATCHES "OpenMP")
    message ("  OpenMP found and linking")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(lapackpp PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# User did not provide a LAPACK Library but specified to search for one
if(USE_OPTIMIZED_LAPACK)
    find_package(LAPACK)
endif()

# Check the usage of the user provided or automatically found LAPACK libraries
if(LAPACK_LIBRARIES)
    include(CheckFortranFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    # Check if new routine of 3.4.0 is in LAPACK_LIBRARIES
    CHECK_FORTRAN_FUNCTION_EXISTS("dgeqrt" LATESTLAPACK_FOUND)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(LATESTLAPACK_FOUND)
        message(STATUS "--> LAPACK supplied by user is WORKING, will use ${LAPACK_LIBRARIES}.")
    else()
        message(ERROR "--> LAPACK supplied by user is not WORKING or is older than LAPACK 3.4.0, CANNOT USE ${LAPACK_LIBRARIES}.")
        message(ERROR "-->     Will use REFERENCE LAPACK (by default)")
        message(ERROR "-->     Or Correct your LAPACK_LIBRARIES entry ")
        message(ERROR "-->     Or Consider checking USE_OPTIMIZED_LAPACK")
    endif()
endif()

include(cmake/LAPACKConfig.cmake)

target_include_directories(lapackpp
    PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(lapackpp
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# Some debug
if(LAPACKPP_QUIET)
    message("MKL defines: " ${MKL_DEFINES})
    message("BLAS int defines: " ${BLAS_int})
    message("FORTRAN MANGLING defines: " ${FORTRAN_MANGLING_DEFINES})
    message("BLAS defines: " ${BLAS_defines})
    message("CBLAS DEFINES: " ${CBLAS_DEFINES})
    message("BLAS RETURN: " ${BLAS_RETURN})
    message("LAPACK DEFINES: " ${LAPACK_DEFINES})
    message("LAPACKE DEFINES: " ${LAPACKE_DEFINES})
    message("XBLAS DEFINES: " ${XBLAS_DEFINES})
    message("LAPACK version define: " ${LAPACK_VER_DEFINE})
endif()

target_compile_definitions(lapackpp PUBLIC
    # Example definition set:
    # -DFORTRAN_ADD_ -DADD_ -DHAVE_BLAS -DBLAS_COMPLEX_RETURN_ARGUMENT -DHAVE_MKL -DHAVE_CBLAS -DHAVE_LAPACK -DLAPACK_VERSION=30700 -DHAVE_XBLAS -DHAVE_LAPACKE
    ${FORTRAN_MANGLING_DEFINES}
    ${BLAS_defines}
    ${BLAS_RETURN}
    ${MKL_DEFINES}
    ${CBLAS_DEFINES}
    ${LAPACK_DEFINES}
    ${BLAS_int}
    ${LAPACK_VER_DEFINE}
    ${XBLAS_DEFINES}
    #${LAPACKE_DEFINES}
)

# Set default install directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/slate"
        CACHE PATH
        "Install path prefix, prepended onto install directories."
        FORCE
    )
endif()

install (TARGETS lapackpp
    EXPORT lapackppTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

configure_file(
    lapackppConfig.cmake.in
    lapackppConfig.cmake
    COPYONLY
)

export(EXPORT lapackppTargets
    FILE lapackppTargets.cmake
)

install (EXPORT lapackppTargets
    FILE lapackppTargets.cmake
    DESTINATION lib/lapackpp
)

install (
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/lapackppConfig.cmake
    DESTINATION lib/lapackpp
)

install(
    DIRECTORY include/
    DESTINATION include
)

# Custom target to mimic makefile target
add_custom_target(lib DEPENDS lapackpp)

if ("${LAPACKE_DEFINES}" MATCHES "HAVE_LAPACKE" AND BUILD_LAPACKPP_TESTS)
    message(STATUS "LAPACKPP testers will be built.")
    add_subdirectory(test)
endif()

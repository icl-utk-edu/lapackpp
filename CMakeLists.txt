# CMake script for LAPACK++ library
# repo: http://bitbucket.org/icl/lapackpp
# Requires BLAS++ library from
#     http://bitbucket.org/icl/blaspp
# Tests require TestSweeper library from
#     http://bitbucket.org/icl/testsweeper

cmake_minimum_required( VERSION 3.15 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_11 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional
# 3.15 message DEBUG, string REPEAT

project(
    lapackpp
    VERSION 2020.06.00
    LANGUAGES CXX
)

# When built as a sub-project, add a namespace to make targets unique,
# e.g., `make tester` becomes `make lapackpp_tester`.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set( lapackpp_is_project true )
    set( lapackpp_ "" )
else()
    set( lapackpp_is_project false )
    set( lapackpp_ "lapackpp_" )
endif()

#-------------------------------------------------------------------------------
# Options
option( BUILD_SHARED_LIBS "Build shared libraries" true )
option( build_tests "Build test suite" "${lapackpp_is_project}" )
option( color "Use ANSI color output" true )
option( use_cmake_find_lapack "Use CMake's find_package( LAPACK ) rather than the search in LAPACK++" false )
option( use_openmp "Use OpenMP, if available" true )

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND lapackpp_is_project)

    set( CMAKE_INSTALL_PREFIX "/opt/slate"
         CACHE PATH
         "Install path prefix, prepended onto install directories."
         FORCE
    )
    message( STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
    # Append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    # This helps find TestSweeper.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
else()
    message( STATUS "Using CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

# Provide menu of options.
set( BLA_VENDOR "" CACHE STRING
     "BLAS Vendor for use in CMake's FindLAPACK. If empty, use LAPACK++ search. Some obsolete options are omitted here." )
set_property(
    CACHE BLA_VENDOR PROPERTY STRINGS
    "" All Goto OpenBLAS FLAME ATLAS IBMESSL
    Intel10_32 Intel10_64lp Intel10_64lp_seq Intel10_64ilp Intel10_64ilp_seq
    Intel10_64_dyn Apple NAS Arm Arm_mp Arm_ilp64 Arm_ilp64_mp Generic )

#-----------------------------------
# LAPACK options
# todo: FLAME, others?
set( lapack "auto" CACHE STRING
     "LAPACK library to search for. Often, LAPACK is included in the BLAS library (e.g., -lopenblas contains both)." )
set_property(
    CACHE lapack PROPERTY STRINGS
    "auto" "generic" )

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling LAPACK++ with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
# Build library.
add_library(
    lapackpp
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
    src/hbev_2stage.cc
    src/hbev.cc
    src/hbevd_2stage.cc
    src/hbevd.cc
    src/hbevx_2stage.cc
    src/hbevx.cc
    src/hbgst.cc
    src/hbgv.cc
    src/hbgvd.cc
    src/hbgvx.cc
    src/hbtrd.cc
    src/hecon_rk.cc
    src/hecon.cc
    src/heequb.cc
    src/heev_2stage.cc
    src/heev.cc
    src/heevd_2stage.cc
    src/heevd.cc
    src/heevr_2stage.cc
    src/heevr.cc
    src/heevx_2stage.cc
    src/heevx.cc
    src/hegst.cc
    src/hegv_2stage.cc
    src/hegv.cc
    src/hegvd.cc
    src/hegvx.cc
    src/herfs.cc
    src/herfsx.cc
    src/hesv_aa.cc
    src/hesv_rk.cc
    src/hesv_rook.cc
    src/hesv.cc
    src/hesvx.cc
    src/heswapr.cc
    src/hetrd_2stage.cc
    src/hetrd.cc
    src/hetrf_aa.cc
    src/hetrf_rk.cc
    src/hetrf_rook.cc
    src/hetrf.cc
    src/hetri_rk.cc
    src/hetri.cc
    src/hetri2.cc
    src/hetrs_aa.cc
    src/hetrs_rk.cc
    src/hetrs_rook.cc
    src/hetrs.cc
    src/hetrs2.cc
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
    src/larfy.cc
    src/larfy.cc
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
    src/sbev_2stage.cc
    src/sbev.cc
    src/sbevd_2stage.cc
    src/sbevd.cc
    src/sbevx_2stage.cc
    src/sbevx.cc
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
    src/sycon_rk.cc
    src/sycon.cc
    src/syequb.cc
    src/syev_2stage.cc
    src/syev.cc
    src/syevd_2stage.cc
    src/syevd.cc
    src/syevr_2stage.cc
    src/syevr.cc
    src/syevx_2stage.cc
    src/syevx.cc
    src/sygst.cc
    src/sygv_2stage.cc
    src/sygv.cc
    src/sygvd.cc
    src/sygvx.cc
    src/symv.cc
    src/syr.cc
    src/syrfs.cc
    src/syrfsx.cc
    src/sysv_aa.cc
    src/sysv_rk.cc
    src/sysv_rook.cc
    src/sysv.cc
    src/sysvx.cc
    src/syswapr.cc
    src/sytrd_2stage.cc
    src/sytrd.cc
    src/sytrf_aa.cc
    src/sytrf_rk.cc
    src/sytrf_rook.cc
    src/sytrf.cc
    src/sytri_rk.cc
    src/sytri.cc
    src/sytri2.cc
    src/sytrs_aa.cc
    src/sytrs_rk.cc
    src/sytrs_rook.cc
    src/sytrs.cc
    src/sytrs2.cc
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
    src/tplqt.cc
    src/tplqt2.cc
    src/tpmlqt.cc
    src/tpmqrt.cc
    src/tpqrt.cc
    src/tpqrt2.cc
    src/tprfb.cc
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
    src/version.cc
)

# Include directory.
# During build it's {source}/include; after install it's {prefix}/include.
target_include_directories(
    lapackpp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

# OpenMP support.
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        set( openmp_lib "OpenMP::OpenMP_CXX" )
        target_link_libraries( lapackpp PUBLIC "${openmp_lib}" )
    endif()
endif()

# Get git commit id.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process( COMMAND git rev-parse --short HEAD
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     OUTPUT_VARIABLE lapackpp_id )
    string( STRIP "${lapackpp_id}" lapackpp_id )
    message( STATUS "lapackpp_id = ${lapackpp_id}" )
    target_compile_definitions(
        lapackpp PRIVATE LAPACKPP_ID="${lapackpp_id}" )
endif()

# Use and export -std=c++11; don't allow -std=gnu++11 extensions.
target_compile_features( lapackpp PUBLIC cxx_std_11 )
set_target_properties( lapackpp PROPERTIES CXX_EXTENSIONS false )

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        lapackpp PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

#-------------------------------------------------------------------------------
# Search for BLAS library.
find_package( blaspp )
if (blaspp_FOUND)
    message( STATUS "Found BLAS++: ${blaspp_DIR}" )
else()
    message( FATAL_ERROR "BLAS++ not found. LAPACK++ requires BLAS++ to be installed first." )
endif()

# Search for LAPACK library.
if (BLA_VENDOR OR use_cmake_find_lapack)
    message( DEBUG "Using CMake's FindLAPACK" )
    find_package( LAPACK )
else()
    message( DEBUG "Using LAPACKFinder" )
    include( "cmake/LAPACKFinder.cmake" )
endif()

if (NOT LAPACK_FOUND)
    message( FATAL_ERROR "LAPACK++ requires a LAPACK library and none was found."
             " Ensure that it is accessible in environment variables"
             " $CPATH, $LIBRARY_PATH, and $LD_LIBRARY_PATH." )
endif()

# todo: lapacke only needed for tester.
include( "cmake/LAPACKConfig.cmake" )

# LAPACK_LIBRARIES could be private, but then if an application directly
# calls blas, cblas, lapack, lapacke, mkl, essl, etc., it would need to
# devine the exact same LAPACK_LIBRARIES. For example, the tester calls
# lapacke. Instead, make it public.
target_link_libraries( lapackpp PUBLIC "${LAPACK_LIBRARIES}" )
target_link_libraries( lapackpp PUBLIC blaspp )

target_compile_definitions(
    lapackpp PRIVATE "${blaspp_defines}" "${blaspp_config_defines}"
                     "${lapack_defines}" "${lapack_config_defines}" )

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# Add 'make lib' target.
if (lapackpp_is_project)
    add_custom_target( lib DEPENDS lapackpp )
endif()

#-------------------------------------------------------------------------------
# GNU Filesystem Conventions
include( GNUInstallDirs )
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/lapackpp )

# Install library and add to <package>Targets.cmake
install(
    TARGETS lapackpp
    EXPORT lapackppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  # no default before 3.14
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install header files
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/    # / copies contents, not directory itself
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING REGEX "\\.(h|hh)$"
)

# Install <package>Targets.cmake
install(
    EXPORT lapackppTargets
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Also export <package>Targets.cmake in build directory
export(
    EXPORT lapackppTargets
    FILE "lapackppTargets.cmake"
)

# Install <package>Config.cmake and <package>ConfigVersion.cmake,
# to enable find_package( <package> ).
include( CMakePackageConfigHelpers )
configure_package_config_file(
    "lapackppConfig.cmake.in"
    "lapackppConfig.cmake"
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
write_basic_package_version_file(
    "lapackppConfigVersion.cmake"
    VERSION "${lapackpp_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/lapackppConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/lapackppConfigVersion.cmake"
    DESTINATION ${INSTALL_CONFIGDIR}
)

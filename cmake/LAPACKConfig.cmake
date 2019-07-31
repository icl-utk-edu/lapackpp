string(ASCII 27 Esc)
set(Red         "${Esc}[31m")
set(Blue        "${Esc}[34m")
set(ColourReset "${Esc}[m")

set(local_mangling "-D${FORTRAN_MANGLING_DEFINES}")
set(local_int "-D${BLAS_INT_DEFINES}")

if(NOT "${MKL_DEFINES}" STREQUAL "")
    set(local_mkl_defines "-D${MKL_DEFINES}")
else()
    set(local_mkl_defines "")
endif()
if(NOT "${BLAS_DEFINES}" STREQUAL "")
    set(local_blas_defines "-D${BLAS_DEFINES}")
else()
    set(local_blas_defines "")
endif()
if(NOT "${BLAS_INT_DEFINES}" STREQUAL "")
    set(local_int "-D${BLAS_INT_DEFINES}")
else()
    set(local_int "")
endif()

message ("blas_links: ${BLAS_links}")
message ("blas_cxx_flags: ${BLAS_cxx_flags}")

message(STATUS "Checking for LAPACK POTRF...")

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output1
)

#message ("compile result: ${compile_res1}")
#message ("run result: ${run_res1}")
#message ("compile output: ${compile_output1}")
#message ("run output: ${run_output1}")

# if it compiled and ran, then LAPACK is available
if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  Found LAPACK${ColourReset}")
    set(LAPACK_DEFINES "HAVE_LAPACK")
else()
    message("${Red}  LAPACK not found${ColourReset}")
    message(STATUS "Checking for separate LAPACK library...")

    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc
        LINK_LIBRARIES
            "-llapack"
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${local_mkl_defines}
            ${local_blas_defines}
            ${local_int}
        COMPILE_OUTPUT_VARIABLE
            compile_OUTPUT1
        RUN_OUTPUT_VARIABLE
            run_output1
        )

    if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        message("${Blue}  Found LAPACK${ColourReset}")
        set(LAPACK_DEFINES "HAVE_LAPACK")
        # Append '-llapack' to BLAS_links
        string(APPEND BLAS_links "-llapack")
    else()
        message("${Red}  LAPACK not found${ColourReset}")
        set(LAPACK_DEFINES "")
    endif()
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

message(STATUS "Checking for LAPACKE POTRF...")

try_run(run_res1 compile_res1
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapacke_potrf.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
)

if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  Found LAPACKE${ColourReset}")
    set(LAPACKE_DEFINES "HAVE_LAPACKE")
else()
    #message("${Red}  LAPACKE was not found${ColourReset}")
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(LAPACKE_DEFINES "")

    find_package (LAPACKE)
    #message ("lapacke_found:        ${LAPACKE_FOUND}")
    #message ("lapacke_libraries:    ${LAPACKE_LIBRARIES}")
    #message ("lapacke_include_dirs: ${LAPACKE_INCLUDE_DIRS}")

    try_run(run_res1 compile_res1
        ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/lapacke_potrf.cc
        LINK_LIBRARIES
            "-l${LAPACKE_LIBRARIES}"
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${local_mkl_defines}
            ${local_blas_defines}
            ${local_int}
            "-I${LAPACKE_INCLUDE_DIRS}"
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
        )

    #message ('compile result: ' ${compile_res1})
    #message ('run result: ' ${run_res1})
    #message ('compile output: ' ${compile_output1})
    #message ('run output: ' ${run_output1})

    if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        message("${Blue}  Found LAPACKE${ColourReset}")
        set(LAPACKE_DEFINES "HAVE_LAPACKE")
    else()
        message("${Red}  LAPACKE was not found${ColourReset}")
    endif()
endif()
set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

message(STATUS "Checking for XBLAS...")

try_run(run_res1 compile_res1
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_xblas.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
)

if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  Found XBLAS${ColourReset}")
    set(XBLAS_DEFINES "HAVE_XBLAS")
else()
    message("${Red}  XBLAS not found.${ColourReset}")
    set(XBLAS_DEFINES "")
endif()
set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

message(STATUS "Checking LAPACK version...")

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_version.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
)

if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  Found LAPACK version number.${ColourReset}")

    #message("run_output1: " ${run_output1})
    string(REPLACE "=" ";" run_out_list ${run_output1})
    #message("run_out_list: " ${run_out_list})
    list(LENGTH run_out_list len)
    #message("len = ${len}")
    list(GET run_out_list 1 version_number)
    #message("version_number: " ${version_number})
    string(REPLACE "." ";" version_list ${version_number})
    #message("version_list: " ${version_list})

    list(GET version_list 0 major_ver)
    list(GET version_list 1 minor_ver)
    list(GET version_list 2 rev_ver)

    # For some reason, the version number strings have extra characters, remove.
    string(REGEX REPLACE "[^0-9]" "" minor_ver ${minor_ver})
    string(LENGTH ${minor_ver} minor_len)
    if(minor_len LESS 2)
        set(minor_ver "0${minor_ver}")
    endif()

    # Remove extra characters.
    string(REGEX REPLACE "[^0-9]" "" rev_ver ${rev_ver})
    string(LENGTH ${rev_ver} rev_len)
    if(rev_len LESS 2)
        set(rev_ver "0${rev_ver}")
    endif()

    set(LAPACK_VER_DEFINE "LAPACK_VERSION=${major_ver}${minor_ver}${rev_ver}")
    message("${Blue}  ${LAPACK_VER_DEFINE}${ColourReset}")
else()
    message("${Red}  Failed to determine LAPACK version.${ColourReset}")
    set(LAPACK_VER_DEFINE "")
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

#message("lapack defines: " ${LAPACK_DEFINES})
#message("lapacke defines: " ${LAPACKE_DEFINES})
#message("xblas defines: " ${XBLAS_DEFINES})
#message("lapack version define: " ${LAPACK_VER_DEFINE})

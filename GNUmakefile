# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# See INSTALL.md for usage.

#-------------------------------------------------------------------------------
# Configuration
# Variables defined in make.inc, or use make's defaults:
#   CXX, CXXFLAGS   -- C compiler and flags
#   LD, LDFLAGS, LIBS -- Linker, options, library paths, and libraries
#   AR, RANLIB      -- Archiver, ranlib updates library TOC
#   prefix          -- where to install LAPACK++

ifeq (${MAKECMDGOALS},config)
    # For `make config`, don't include make.inc with previous config;
    # force re-creating make.inc.
    .PHONY: config
    config: make.inc

    make.inc: force
else ifneq (clean,${findstring clean,${MAKECMDGOALS}})
    # For `make clean` or `make distclean`, don't include make.inc,
    # which could generate it. Otherwise, include make.inc.
    include make.inc
endif

python ?= python3

force: ;

make.inc:
	${python} configure.py

# Defaults if not given in make.inc. GNU make doesn't have defaults for these.
RANLIB   ?= ranlib
prefix   ?= /opt/slate

abs_prefix := ${abspath ${prefix}}

# Default LD=ld won't work; use CXX. Can override in make.inc or environment.
ifeq (${origin LD},default)
    LD = ${CXX}
endif

# Use abi-compliance-checker to compare the ABI (application binary
# interface) of 2 releases. Changing the ABI does not necessarily change
# the API (application programming interface). Rearranging a struct or
# changing a by-value argument from int64 to int doesn't change the
# API--no source code changes are required, just a recompile.
#
# if structs or routines are changed or removed:
#     bump major version and reset minor, revision = 0;
# else if structs or routines are added:
#     bump minor version and reset revision = 0;
# else (e.g., bug fixes):
#     bump revision
#
# soversion is major ABI version.
abi_version = 2.0.0
soversion = ${word 1, ${subst ., ,${abi_version}}}

#-------------------------------------------------------------------------------
ldflags_shared = -shared

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype := ${shell echo $${OSTYPE}}
ifneq (,${findstring darwin, ${ostype}})
    # MacOS is darwin
    macos = 1
    # MacOS needs shared library's path set, and shared library version.
    ldflags_shared += -install_name @rpath/${notdir $@} \
                      -current_version ${abi_version} \
                      -compatibility_version ${soversion}
    so = dylib
    so2 = .dylib
    # on macOS, .dylib comes after version: libfoo.4.dylib
else
    # Linux needs shared library's soname.
    ldflags_shared += -Wl,-soname,${notdir ${lib_soname}}
    so = so
    so1 = .so
    # on Linux, .so comes before version: libfoo.so.4
endif

#-------------------------------------------------------------------------------
# if shared
ifneq (${static},1)
    CXXFLAGS += -fPIC
    LDFLAGS  += -fPIC
    lib_ext = ${so}
else
    lib_ext = a
endif

#-------------------------------------------------------------------------------
# Files

lib_src  = ${wildcard src/*.cc src/cuda/*.cc src/rocm/*.cc src/onemkl/*.cc src/stub/*.cc}
lib_obj  = ${addsuffix .o, ${basename ${lib_src}}}
dep     += ${addsuffix .d, ${basename ${lib_src}}}

tester_src = ${wildcard test/*.cc}
tester_obj = ${addsuffix .o, ${basename ${tester_src}}}
dep       += ${addsuffix .d, ${basename ${tester_src}}}

tester = test/tester

pkg = lib/pkgconfig/lapackpp.pc

#-------------------------------------------------------------------------------
# BLAS++
# todo: should configure.py save blaspp_dir & testsweeper_dir in make.inc?
# Order here (./blaspp, ../blaspp) is reverse of order in configure.py.

blaspp_dir = ${wildcard ./blaspp}
ifeq (${blaspp_dir},)
    blaspp_dir = ${wildcard ../blaspp}
endif

blaspp_src = ${wildcard ${blaspp_dir}/src/*.cc ${blaspp_dir}/include/*.hh}

blaspp = ${blaspp_dir}/lib/libblaspp.${lib_ext}

blaspp: ${blaspp}

ifneq (${blaspp_dir},)
    ${blaspp}: ${blaspp_src}
		cd ${blaspp_dir} && ${MAKE} lib CXX=${CXX}
else
    ${blaspp}:
		${error LAPACK++ requires BLAS++, which was not found. Run 'make config' \
		        or download manually from https://github.com/icl-utk-edu/blaspp}
endif

# Compile BLAS++ before LAPACK++.
${lib_obj} ${tester_obj}: | ${blaspp}


#-------------------------------------------------------------------------------
# TestSweeper

testsweeper_dir = ${wildcard ../testsweeper}
ifeq (${testsweeper_dir},)
    testsweeper_dir = ${wildcard ${blaspp_dir}/testsweeper}
endif
ifeq (${testsweeper_dir},)
    testsweeper_dir = ${wildcard ./testsweeper}
endif

testsweeper_src = ${wildcard ${testsweeper_dir}/testsweeper.cc ${testsweeper_dir}/testsweeper.hh}

testsweeper = ${testsweeper_dir}/libtestsweeper.${lib_ext}

testsweeper: ${testsweeper}

ifneq (${testsweeper_dir},)
    ${testsweeper}: ${testsweeper_src}
		cd ${testsweeper_dir} && ${MAKE} lib CXX=${CXX}
else
    ${testsweeper}:
		${error Tester requires TestSweeper, which was not found. Run 'make config' \
		        or download manually from https://github.com/icl-utk-edu/testsweeper}
endif

# Compile TestSweeper before LAPACK++.
${lib_obj} ${tester_obj}: | ${testsweeper}

#-------------------------------------------------------------------------------
# Get Mercurial id, and make version.o depend on it via .id file.

ifneq (${wildcard .git},)
    id := ${shell git rev-parse --short HEAD}
    src/version.o: CXXFLAGS += -DLAPACKPP_ID='"${id}"'
endif

last_id := ${shell [ -e .id ] && cat .id || echo 'NA'}
ifneq (${id},${last_id})
    .id: force
endif

.id:
	echo ${id} > .id

src/version.o: .id

#-------------------------------------------------------------------------------
# LAPACK++ specific flags and libraries
CXXFLAGS += -I./include
CXXFLAGS += -I${blaspp_dir}/include

# additional flags and libraries for testers
${tester_obj}: CXXFLAGS += -I${testsweeper_dir}

TEST_LDFLAGS += -L./lib -Wl,-rpath,${abspath ./lib}
TEST_LDFLAGS += -L${blaspp_dir}/lib -Wl,-rpath,${abspath ${blaspp_dir}/lib}
TEST_LDFLAGS += -L${testsweeper_dir} -Wl,-rpath,${abspath ${testsweeper_dir}}
TEST_LIBS    += -llapackpp -lblaspp -ltestsweeper

#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all docs hooks lib src test tester headers include clean distclean
.DEFAULT_GOAL = all

all: lib tester hooks

install: lib ${pkg}
	mkdir -p ${DESTDIR}${abs_prefix}/include/lapack
	mkdir -p ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig
	cp include/*.hh        ${DESTDIR}${abs_prefix}/include/
	cp include/lapack/*.h  ${DESTDIR}${abs_prefix}/include/lapack/
	cp include/lapack/*.hh ${DESTDIR}${abs_prefix}/include/lapack/
	cp -av ${lib_name}*    ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/
	cp ${pkg}              ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig/
	cd ${blaspp_dir} && make install prefix=${prefix}

uninstall:
	${RM}    ${DESTDIR}${abs_prefix}/include/lapack.hh
	${RM} -r ${DESTDIR}${abs_prefix}/include/lapack
	${RM}    ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/${notdir ${lib_name}*}
	${RM}    ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig/lapackpp.pc

#-------------------------------------------------------------------------------
# if re-configured, recompile everything
${lib_obj} ${tester_obj}: make.inc

#-------------------------------------------------------------------------------
# Generic rule for shared libraries.
# For libfoo.so version 4.5.6, this creates libfoo.so.4.5.6 and symlinks
# libfoo.so.4 -> libfoo.so.4.5.6
# libfoo.so   -> libfoo.so.4
#
# Needs [private] variables set (shown with example values):
# LDFLAGS     = -L/path/to/lib
# LIBS        = -lmylib
# lib_obj     = src/foo.o src/bar.o
# lib_so_abi  = libfoo.so.4.5.6
# lib_soname  = libfoo.so.4
# abi_version = 4.5.6
# soversion   = 4
%.${lib_ext}:
	mkdir -p lib
	${LD} ${LDFLAGS} ${ldflags_shared} ${LIBS} ${lib_obj} -o ${lib_so_abi}
	ln -fs ${notdir ${lib_so_abi}} ${lib_soname}
	ln -fs ${notdir ${lib_soname}} $@

# Generic rule for static libraries, creates libfoo.a.
# The library should depend only on its objects.
%.a:
	mkdir -p lib
	${RM} $@
	${AR} cr $@ $^
	${RANLIB} $@

#-------------------------------------------------------------------------------
# LAPACK++ library
# so     is like libfoo.so       or libfoo.dylib
# so_abi is like libfoo.so.4.5.6 or libfoo.4.5.6.dylib
# soname is like libfoo.so.4     or libfoo.4.dylib
lib_name   = lib/liblapackpp
lib_a      = ${lib_name}.a
lib_so     = ${lib_name}.${so}
lib        = ${lib_name}.${lib_ext}
lib_so_abi = ${lib_name}${so1}.${abi_version}${so2}
lib_soname = ${lib_name}${so1}.${soversion}${so2}

${lib_so}: ${lib_obj}

${lib_a}: ${lib_obj}

# sub-directory rules
lib src: ${lib}

lib/clean src/clean:
	${RM} ${lib_a} ${lib_so} ${lib_so_abi} ${lib_soname} ${lib_obj}

#-------------------------------------------------------------------------------
# tester
${tester}: ${tester_obj} ${lib} ${testsweeper} ${blaspp}
	${LD} ${TEST_LDFLAGS} ${LDFLAGS} ${tester_obj} \
		${TEST_LIBS} ${LIBS} -o $@

# sub-directory rules
# Note 'test' is sub-directory rule; 'tester' is CMake-compatible rule.
test: ${tester}
tester: ${tester}

test/clean:
	${RM} ${tester} test/*.o

test/check: check

# 'make check' tests subset of routines, to avoid spurious failures
check: tester
	cd test; ${python} run_tests.py --quick \
		gesv getrf posv potrf geqrf ungqr gels \
		geev heev heevd heevr gesvd

#-------------------------------------------------------------------------------
# headers
# precompile headers to verify self-sufficiency
headers     = ${wildcard include/lapack.hh include/lapack/*.h include/lapack/*.hh test/*.hh}
headers_gch = ${addsuffix .gch, ${basename ${headers}}}

headers: ${headers_gch}

headers/clean:
	${RM} ${headers_gch}

# sub-directory rules
include: headers

include/clean: headers/clean

#-------------------------------------------------------------------------------
# pkgconfig
# Keep -std=c++11 in CXXFLAGS. Keep -fopenmp in LDFLAGS.
CXXFLAGS_clean = ${filter-out -O% -W% -pedantic -D% -I./include -I${blaspp_dir}% -MMD -fPIC -fopenmp, ${CXXFLAGS}}
CPPFLAGS_clean = ${filter-out -O% -W% -pedantic -D% -I./include -I${blaspp_dir}% -MMD -fPIC -fopenmp, ${CPPFLAGS}}
LDFLAGS_clean  = ${filter-out -fPIC, ${LDFLAGS}}

.PHONY: ${pkg}
${pkg}:
	perl -pe "s'#VERSION'2024.10.26'; \
	          s'#PREFIX'${abs_prefix}'; \
	          s'#CXX\b'${CXX}'; \
	          s'#CXXFLAGS'${CXXFLAGS_clean}'; \
	          s'#CPPFLAGS'${CPPFLAGS_clean}'; \
	          s'#LDFLAGS'${LDFLAGS_clean}'; \
	          s'#LIBS'${LIBS}';" \
	          $@.in > $@

#-------------------------------------------------------------------------------
# documentation
docs: docs/html/index.html

doc_files = \
	docs/doxygen/DoxygenLayout.xml \
	docs/doxygen/doxyfile.conf \
	docs/doxygen/groups.dox \
	README.md \
	INSTALL.md \

docs/html/index.html: ${headers} ${lib_src} ${tester_src} ${doc_files}
	doxygen docs/doxygen/doxyfile.conf
	@echo ========================================
	cat docs/doxygen/errors.txt
	@echo ========================================
	@echo "Documentation available in docs/html/index.html"
	@echo ========================================

# sub-directory redirects
src/docs: docs
test/docs: docs

#-------------------------------------------------------------------------------
# general rules
clean: lib/clean test/clean headers/clean
	${RM} ${dep}

distclean: clean
	${RM} make.inc include/lapack/defines.h

# Install git hooks
hooks = .git/hooks/pre-commit

hooks: ${hooks}

.git/hooks/%: tools/hooks/%
	@if [ -e .git/hooks ]; then \
		echo cp $< $@ ; \
		cp $< $@ ; \
	fi

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $< -o $@

# preprocess source
%.i: %.cc
	${CXX} ${CXXFLAGS} -I${blaspp_dir}/test -I${testsweeper_dir} -E $< -o $@

# preprocess source
%.i: %.h
	${CXX} ${CXXFLAGS} -I${blaspp_dir}/test -I${testsweeper_dir} -E $< -o $@

# preprocess source
%.i: %.hh
	${CXX} ${CXXFLAGS} -I${blaspp_dir}/test -I${testsweeper_dir} -E $< -o $@

# precompile header to check for errors
%.gch: %.h
	${CXX} ${CXXFLAGS} -I${blaspp_dir}/test -I${testsweeper_dir} -c $< -o $@

%.gch: %.hh
	${CXX} ${CXXFLAGS} -I${blaspp_dir}/test -I${testsweeper_dir} -c $< -o $@

-include ${dep}

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "---------- Options"
	@echo "static        = '${static}'"
	@echo "prefix        = '${prefix}'"
	@echo "abs_prefix    = '${abs_prefix}'"
	@echo
	@echo "---------- Internal variables"
	@echo "ostype        = '${ostype}'"
	@echo "macos         = '${macos}'"
	@echo "id            = '${id}'"
	@echo "last_id       = '${last_id}'"
	@echo "abi_version   = '${abi_version}'"
	@echo "soversion     = '${soversion}'"
	@echo
	@echo "---------- Libraries"
	@echo "lib_name      = ${lib_name}"
	@echo "lib_a         = ${lib_a}"
	@echo "lib_so        = ${lib_so}"
	@echo "lib           = ${lib}"
	@echo "lib_so_abi    = ${lib_so_abi}"
	@echo "lib_soname    = ${lib_soname}"
	@echo
	@echo "pkg           = ${pkg}"
	@echo
	@echo "lib_src       = ${lib_src}"
	@echo
	@echo "lib_obj       = ${lib_obj}"
	@echo
	@echo "tester_src    = ${tester_src}"
	@echo
	@echo "tester_obj    = ${tester_obj}"
	@echo
	@echo "tester        = ${tester}"
	@echo
	@echo "dep           = ${dep}"
	@echo
	@echo "testsweeper_dir   = ${testsweeper_dir}"
	@echo "testsweeper_src   = ${testsweeper_src}"
	@echo "testsweeper       = ${testsweeper}"
	@echo
	@echo "blaspp_dir    = ${blaspp_dir}"
	@echo "blaspp_src    = ${blaspp_src}"
	@echo "blaspp        = ${blaspp}"
	@echo
	@echo "---------- C++ compiler"
	@echo "CXX           = ${CXX}"
	@echo "CXXFLAGS      = ${CXXFLAGS}"
	@echo
	@echo "---------- Link flags"
	@echo "LD            = ${LD}"
	@echo "LDFLAGS       = ${LDFLAGS}"
	@echo "LIBS          = ${LIBS}"
	@echo "ldflags_shared = ${ldflags_shared}"
	@echo
	@echo "TEST_LDFLAGS  = ${TEST_LDFLAGS}"
	@echo "TEST_LIBS     = ${TEST_LIBS}"

include make.inc

# defaults if not defined in make.inc
LDFLAGS  ?= -fPIC -fopenmp
CXXFLAGS ?= -fPIC -fopenmp -MMD -std=c++11 -pedantic \
            -Wall -Wmissing-declarations \
            -Wno-unused-local-typedefs \
            -I${LAPACKDIR}/LAPACKE/include \
            -DLAPACK_VERSION=30800 \
            -DLAPACK_MATGEN
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

LIBS     ?= -L${LAPACKDIR} -llapacke -llapack -lblas

# ------------------------------------------------------------------------------
# LAPACK++ specific flags
pwd = ${shell pwd}
libtest_path = ${realpath ${pwd}/../libtest}
libtest_src  = ${wildcard ${libtest_path}/*.cc} \
               ${wildcard ${libtest_path}/*.hh}
libtest_so   = ${libtest_path}/libtest.so

LAPACKPP_FLAGS = -I../libtest \
                 -I../blaspp/include \
                 -I../blaspp/test \
                 -Iinclude \
                 -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP

LAPACKPP_LIBS  = -L../libtest -Wl,-rpath,${libtest_path} -ltest \
                 -Llib -Wl,-rpath,${pwd}/lib -llapackpp

# ------------------------------------------------------------------------------
# files
src = ${wildcard src/*.cc}
obj = ${addsuffix .o, ${basename ${src}}}
dep = ${addsuffix .d, ${basename ${src}}}

test_src = ${wildcard test/*.cc}
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

tools_src = ${wildcard tools/*.cc}
tools_obj = ${addsuffix .o, ${basename ${tools_src}}}
tools_dep = ${addsuffix .d, ${basename ${tools_src}}}

liblapackpp_so = lib/liblapackpp.so
liblapackpp_a  = lib/liblapackpp.a

# ------------------------------------------------------------------------------
# MacOS likes shared library's path to be set; see make.inc.macos
ifneq (${INSTALL_NAME},)
    ${liblapackpp_so}: LDFLAGS += ${INSTALL_NAME} @rpath/${notdir ${liblapackpp_so}}
endif

# ------------------------------------------------------------------------------
# rules
.PHONY: default all shared static include src test docs clean test_headers

default: shared test

all: shared static test

shared: ${liblapackpp_so}

static: ${liblapackpp_a}

lib:
	mkdir lib

# defalut rules for subdirectories
include: test_headers

src: shared

test: test/test

tools: tools/lapack_version

docs:
	doxygen docs/doxygen/doxyfile.conf
	@echo ========================================
	cat docs/doxygen/errors.txt
	@echo ========================================
	@echo "Documentation available in docs/html/index.html"
	@echo ========================================

test/test: ${test_obj} ${liblapackpp_so} ${libtest_so}
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${LAPACKPP_LIBS} ${LIBS}

tools/lapack_version: tools/lapack_version.o
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

${liblapackpp_so}: ${obj} | lib
	${CXX} ${LDFLAGS} -shared -o $@ ${obj} ${LIBS}

${liblapackpp_a}: ${obj} | lib
	ar cr $@ ${obj}
	ranlib $@

${libtest_so}: ${libtest_src}
	cd ${libtest_path} && ${MAKE}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -E -o $@ $<

clean: include/clean src/clean test/clean tools/clean

include/clean:
	-${RM} gch/include/*.gch

src/clean:
	-${RM} lib/*.{a,so} src/*.{o,d}

test/clean:
	-${RM} test/test test/*.{o,d} gch/test/*.gch

tools/clean:
	-${RM} tools/lapack_version tools/*.{o,d}

-include ${dep} ${test_dep}

# ------------------------------------------------------------------------------
# subdirectory redirects
src/test_headers: test_headers
test/test_headers: test_headers

src/docs: docs
test/docs: docs

# ------------------------------------------------------------------------------
# precompile headers to verify self-sufficiency
headers     = ${wildcard include/*.h include/*.hh test/*.hh}
headers_gch = ${addprefix gch/, ${addsuffix .gch, ${headers}}}

test_headers: ${headers_gch}

gch/include/%.h.gch: include/%.h | gch/include
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -c -o $@ $<

gch/include/%.hh.gch: include/%.hh | gch/include
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -c -o $@ $<

gch/test/%.hh.gch: test/%.hh | gch/test
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -c -o $@ $<

# make directories
gch/include: | gch
gch/test:    | gch
gch/include gch/test gch:
	mkdir $@

include make.inc

# defaults if not defined in make.inc
CXX      ?= g++

LDFLAGS  ?= -fPIC -fopenmp
CXXFLAGS ?= -fPIC -fopenmp -MMD -std=c++11 -pedantic \
            -Wall -Wmissing-declarations \
            -Wno-unused-local-typedefs \
            -I${LAPACKDIR}/LAPACKE/include \
            -DLAPACK_VERSION_MAJOR=3 -DLAPACK_VERSION_MINOR=8 -DLAPACK_VERSION_MICRO=0 \
            -DLAPACK_MATGEN
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

LIBS     ?= -L${LAPACKDIR} -llapacke -llapack -lblas

# ------------------------------------------------------------------------------
# LAPACK++ specific flags
pwd = ${shell pwd}

LAPACKPP_FLAGS = -I../libtest \
                 -I../blaspp/include \
                 -I../blaspp/test \
                 -Iinclude \
                 -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP

LAPACKPP_LIBS  = -L../libtest -Wl,-rpath,${pwd}/../libtest -ltest \
                 -Llib -Wl,-rpath,${pwd}/lib -llapackpp

# ------------------------------------------------------------------------------
# files
src = ${wildcard src/*.cc}
obj = ${addsuffix .o, ${basename ${src}}}
dep = ${addsuffix .d, ${basename ${src}}}

test_src = ${wildcard test/*.cc}
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

liblapackpp_so = lib/liblapackpp.so
liblapackpp_a  = lib/liblapackpp.a

# ------------------------------------------------------------------------------
# MacOS likes shared library's path to be set; see make.inc.macos
ifneq (${INSTALL_NAME},)
    ${liblapackpp_so}: LDFLAGS += ${INSTALL_NAME} @rpath/${notdir ${liblapackpp_so}}
endif

# ------------------------------------------------------------------------------
# rules
.PHONY: default all shared static include src test clean test_headers

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

test/test: ${test_obj} ${liblapackpp_so}
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${LAPACKPP_LIBS} ${LIBS}

${liblapackpp_so}: ${obj} | lib
	${CXX} ${LDFLAGS} -shared -o $@ ${obj} ${LIBS}

${liblapackpp_a}: ${obj} | lib
	ar cr $@ ${obj}
	ranlib $@

%.o: %.cc
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${LAPACKPP_FLAGS} -E -o $@ $<

clean:
	-${RM} lib/*.a lib/*.so src/*.o src/*.d test/*.o test/*.d test/test

-include ${dep} ${test_dep}

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

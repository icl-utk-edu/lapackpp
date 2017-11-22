SHELL = /bin/sh

CXX = g++

CXXFLAGS = -g -fopenmp -MMD -Wall -Wno-unused-local-typedefs \
           -I../../blaspp/include \
           -I../../blaspp/test \
           -I../include \
           -I../../libtest \
           -I${LAPACKDIR}/LAPACKE/include

CXXFLAGS += -DLAPACK_COMPLEX_CPP

LDFLAGS  = -fopenmp \
            -Wl,-rpath,../../libtest

LIBS     = -L../lib -llapackpp \
           -L../../libtest -ltest \
           -L${LAPACKDIR} -llapacke \
           -framework Accelerate \

src = ${wildcard *.cc}
obj = ${addsuffix .o, ${basename ${src}}}
dep = ${addsuffix .d, ${basename ${src}}}

libsrc = ${wildcard ../src/*.cc}

.DELETE_ON_ERROR:

.SUFFIXES:

# ------------------------------------------------------------------------------
# Rules

all: test

test: ${obj} ../lib/liblapackpp.so
	${CXX} ${LDFLAGS} -o $@ ${obj} ${LIBS}

../lib/liblapackpp.so: ${libsrc}
	cd ../src && ${MAKE}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c -o $@ $<

# preprocess
%.i: %.cc
	${CXX} ${CXXFLAGS} -E -o $@ $<

clean:
	-rm -f test *.o *.d

-include ${dep}

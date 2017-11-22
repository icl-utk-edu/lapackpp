CXX      = g++
CXXFLAGS = -std=c++11 -O2 -fPIC \
           -Wall -Wmissing-declarations -Wno-unused-local-typedefs \
           -I../../blaspp/include \
           -I../include \
           -DLAPACK_COMPLEX_CPP

# Accelerate missing *gbrfsx, looks like LAPACK 3.1
CXXFLAGS += -DLAPACK_VERSION_MAJOR=3 -DLAPACK_VERSION_MINOR=1 -DLAPACK_VERSION_MICRO=0
#CXXFLAGS += -DLAPACK_MATGEN
LIBS     = -framework Accelerate

src = ${wildcard *.cc}
obj = ${addsuffix .o, ${basename ${src}}}
dep = ${addsuffix .d, ${basename ${src}}}

default: shared

all: shared static

shared: ../lib/liblapackpp.so

static: ../lib/liblapackpp.a

../lib:
	mkdir $@

../lib/liblapackpp.so: ${obj} | ../lib
	${CXX} -shared -o $@ ${obj} ${LIBS}

../lib/liblapackpp.a: ${obj} | ../lib
	ar cr $@ ${obj}
	ranlib $@

echo:
	@echo "src ${src}"
	@echo "obj ${obj}"

%.o: %.cc
	${CXX} ${CXXFLAGS} -c -o $@ $<

clean:
	-rm -f *.o

.PHONY: differ.txt custom.txt

differ.txt:
	@diff --brief ../gen . \
		| perl -ne 'if (m@Files .* \./(\w+\.cc) differ@) { print "$$1\n"; }' \
		> $@

custom.txt:
	@ls -1 custom > $@

diff: differ.txt custom.txt
	diff differ.txt custom.txt

-include ${dep}

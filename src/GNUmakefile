CXX = g++
CXXFLAGS = -Wall -I../../blaspp/include -I../include -std=c++11 -DLAPACK_COMPLEX_CPP -Wno-unused-local-typedefs

src = ${wildcard *.cc}
obj = ${addsuffix .o, ${basename ${src}}}
dep = ${addsuffix .d, ${basename ${src}}}

all: ${obj}

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

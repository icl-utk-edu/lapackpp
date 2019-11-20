# Usage:
# make by default:
#    - Runs configure.py to create make.inc, if it doesn't exist.
#    - Compiles lib/liblapackpp.so, or lib/liblapackpp.a (if static=1).
#    - Compiles the tester, test/tester.
#
# make config    - Runs configure.py to create make.inc.
# make lib       - Compiles lib/liblapackpp.so, or liblapackpp.a (if static=1).
# make tester      - Compiles the tester, test/tester.
# make docs      - Compiles Doxygen documentation.
# make install   - Installs the library and headers to $prefix.
# make clean     - Deletes all objects, libraries, and the tester.
# make distclean - Also deletes make.inc and dependency files (*.d).

#-------------------------------------------------------------------------------
# Configuration
# Variables defined in make.inc, or use make's defaults:
#   CXX, CXXFLAGS   -- C compiler and flags
#   LDFLAGS, LIBS   -- Linker options, library paths, and libraries
#   AR, RANLIB      -- Archiver, ranlib updates library TOC
#   prefix          -- where to install LAPACK++

include make.inc

# Existence of .make.inc.$${PPID} is used so 'make config' doesn't run
# configure.py twice when make.inc doesn't exist initially.
make.inc:
	python configure.py
	touch .make.inc.$${PPID}

.PHONY: config
config:
	if [ ! -e .make.inc.$${PPID} ]; then \
		python configure.py; \
	fi

# defaults if not given in make.inc
CXXFLAGS ?= -O3 -std=c++11 -MMD \
            -Wall -pedantic \
            -Wshadow \
            -Wno-unused-local-typedefs \
            -Wno-unused-function \

#CXXFLAGS += -Wmissing-declarations
#CXXFLAGS += -Wconversion
#CXXFLAGS += -Werror

# GNU make doesn't have defaults for these
RANLIB   ?= ranlib
prefix   ?= /usr/local/lapackpp

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype = $(shell echo $${OSTYPE})
ifneq ($(findstring darwin, $(ostype)),)
	# MacOS is darwin
	macos = 1
endif

#-------------------------------------------------------------------------------
# if shared
ifneq ($(static),1)
	CXXFLAGS += -fPIC
	LDFLAGS  += -fPIC
endif

#-------------------------------------------------------------------------------
# MacOS needs shared library's path set
ifeq ($(macos),1)
	install_name = -install_name @rpath/$(notdir $@)
else
	install_name =
endif

#-------------------------------------------------------------------------------
# Files

lib_src  = $(wildcard src/*.cc)
lib_obj  = $(addsuffix .o, $(basename $(lib_src)))
dep     += $(addsuffix .d, $(basename $(lib_src)))

tester_src = $(wildcard test/*.cc)
tester_obj = $(addsuffix .o, $(basename $(tester_src)))
dep       += $(addsuffix .d, $(basename $(tester_src)))

tester     = test/tester

blaspp_dir = $(wildcard ../blaspp)
ifeq ($(blaspp_dir),)
	blaspp_dir = $(wildcard ./blaspp)
endif
ifeq ($(blaspp_dir),)
    $(lib_obj):
		$(error LAPACK++ requires BLAS++, which was not found. Run 'make config' \
		        or download manually from https://bitbucket.org/icl/blaspp/)
endif

blaspp_src = $(wildcard $(blaspp_dir)/src/*.cc $(blaspp_dir)/include/*.hh)
ifeq ($(static),1)
	libblaspp  = $(blaspp_dir)/lib/libblaspp.a
else
	libblaspp  = $(blaspp_dir)/lib/libblaspp.so
endif

libtest_dir = $(wildcard ../libtest)
ifeq ($(libtest_dir),)
	libtest_dir = $(wildcard $(blaspp_dir)/libtest)
endif
ifeq ($(libtest_dir),)
	libtest_dir = $(wildcard ./libtest)
endif
ifeq ($(libtest_dir),)
    $(tester_obj):
		$(error Tester requires libtest, which was not found. Run 'make config' \
		        or download manually from https://bitbucket.org/icl/libtest/)
endif

libtest_src = $(wildcard $(libtest_dir)/libtest.cc $(libtest_dir)/libtest.hh)
ifeq ($(static),1)
	libtest = $(libtest_dir)/libtest.a
else
	libtest = $(libtest_dir)/libtest.so
endif

lib_a  = ./lib/liblapackpp.a
lib_so = ./lib/liblapackpp.so

ifeq ($(static),1)
	lib = $(lib_a)
else
	lib = $(lib_so)
endif

#-------------------------------------------------------------------------------
# LAPACK++ specific flags and libraries
CXXFLAGS += -I./include
CXXFLAGS += -I$(blaspp_dir)/include

# additional flags and libraries for testers
$(tester_obj): CXXFLAGS += -I$(libtest_dir)

TEST_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
TEST_LDFLAGS += -L$(blaspp_dir)/lib -Wl,-rpath,$(abspath $(blaspp_dir)/lib)
TEST_LDFLAGS += -L$(libtest_dir) -Wl,-rpath,$(abspath $(libtest_dir))
TEST_LIBS    += -lblaspp -llapackpp -ltest

#-------------------------------------------------------------------------------
# Rules

targets = all lib src tester headers include docs clean distclean

.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: $(targets)
.DEFAULT_GOAL = all

all: lib tester

install: lib
	mkdir -p $(DESTDIR)$(prefix)/include
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	cp include/*.{h,hh} $(DESTDIR)$(prefix)/include
	cp lib/liblapackpp.* $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)

uninstall:
	$(RM) $(addprefix $(DESTDIR)$(prefix), $(headers))
	$(RM) $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/liblapackpp.*

#-------------------------------------------------------------------------------
# if re-configured, recompile everything
$(lib_obj) $(tester_obj): lapack_defines.h

#-------------------------------------------------------------------------------
# LAPACK++ library
$(lib_so): $(lib_obj)
	mkdir -p lib
	$(CXX) $(LDFLAGS) -shared $(install_name) $(lib_obj) $(LIBS) -o $@

$(lib_a): $(lib_obj)
	mkdir -p lib
	$(RM) $@
	$(AR) cr $@ $(lib_obj)
	$(RANLIB) $@

# sub-directory rules
lib src: $(lib)

lib/clean src/clean:
	$(RM) lib/*.{a,so} src/*.o

#-------------------------------------------------------------------------------
# BLAS++ library
ifneq ($(blaspp_dir),)
    $(libblaspp): $(libblaspp_src)
		cd $(blaspp_dir) && $(MAKE) lib CXX=$(CXX)
endif

#-------------------------------------------------------------------------------
# libtest library
ifneq ($(libtest_dir),)
    $(libtest): $(libtest_src)
		cd $(libtest_dir) && $(MAKE) lib CXX=$(CXX)
endif

#-------------------------------------------------------------------------------
# tester
$(tester): $(tester_obj) $(lib) $(libtest) $(libblaspp)
	$(CXX) $(TEST_LDFLAGS) $(LDFLAGS) $(tester_obj) \
		$(TEST_LIBS) $(LIBS) -o $@

# sub-directory rules
tester: $(tester)

tester/clean:
	$(RM) $(tester) test/*.o

#-------------------------------------------------------------------------------
# headers
# precompile headers to verify self-sufficiency
headers     = $(wildcard include/*.h include/*.hh test/*.hh)
headers_gch = $(addsuffix .gch, $(headers))

headers: $(headers_gch)

headers/clean:
	$(RM) include/*.h.gch include/*.hh.gch test/*.hh.gch

# sub-directory rules
include: headers

include/clean: headers/clean

#-------------------------------------------------------------------------------
# documentation
docs:
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
clean: lib/clean tester/clean headers/clean

distclean: clean
	$(RM) make.inc src/*.d test/*.d

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# preprocess source
%.i: %.cc
	$(CXX) $(CXXFLAGS) -I$(blaspp_dir)/test -I$(libtest_dir) -E $< -o $@

# precompile header to check for errors
%.h.gch: %.h
	$(CXX) $(CXXFLAGS) -I$(blaspp_dir)/test -I$(libtest_dir) -c $< -o $@

%.hh.gch: %.hh
	$(CXX) $(CXXFLAGS) -I$(blaspp_dir)/test -I$(libtest_dir) -c $< -o $@

-include $(dep)

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "static        = '$(static)'"
	@echo
	@echo "lib_a         = $(lib_a)"
	@echo "lib_so        = $(lib_so)"
	@echo "lib           = $(lib)"
	@echo
	@echo "lib_src       = $(lib_src)"
	@echo
	@echo "lib_obj       = $(lib_obj)"
	@echo
	@echo "tester_src      = $(tester_src)"
	@echo
	@echo "tester_obj      = $(tester_obj)"
	@echo
	@echo "tester          = $(tester)"
	@echo
	@echo "dep           = $(dep)"
	@echo
	@echo "libtest_dir   = $(libtest_dir)"
	@echo "libtest_src   = $(libtest_src)"
	@echo "libtest       = $(libtest)"
	@echo
	@echo "blaspp_dir    = $(blaspp_dir)"
	@echo "blaspp_src    = $(blaspp_src)"
	@echo "libblaspp     = $(libblaspp)"
	@echo
	@echo "CXX           = $(CXX)"
	@echo "CXXFLAGS      = $(CXXFLAGS)"
	@echo
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LIBS          = $(LIBS)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIBS     = $(TEST_LIBS)"

# CXX compiler must match the one used to compiler LAPACK++.
# Set it in your environment or here.
# Sadly, pkg-config doesn't provide a way to query CXX,
# CXXFLAGS (only cflags), CPPFLAGS, or LDFLAGS.

#-------------------------------------------------------------------------------
# Set CXXFLAGS and LIBS
pkg_exists := $(shell pkg-config --exists lapackpp; echo $$?)
ifeq ($(pkg_exists),0)

  # Get flags from pkg-config.
  CXX      = $(shell pkg-config --variable CXX lapackpp)
  CXXFLAGS = $(shell pkg-config --cflags lapackpp)
  LIBS     = $(shell pkg-config --libs   lapackpp)

else
  $(warning WARNING: pkg-config couldn't find lapackpp. Using hard-coded flags in Makefile.)

  # LAPACK++ not in pkg-config.
  # Here's a hard-coded example using OpenBLAS.
  CXXFLAGS = -I/usr/local/lapackpp/include -std=c++11
  LIBS     = -L/usr/local/lapackpp/lib$(LIB_SUFFIX) -llapackpp -lopenblas

endif

#-------------------------------------------------------------------------------
# Rules
example_potrf: example_potrf.o
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	-rm -f example_potrf *.o

#-------------------------------------------------------------------------------
# Debugging
echo:
	@echo "PKG_CONFIG_PATH $(PKG_CONFIG_PATH)"
	@echo "pkg_exists $(pkg_exists)"
	@echo "CXX        $(CXX)"
	@echo "CXXFLAGS   $(CXXFLAGS)"
	@echo "LIBS       $(LIBS)"

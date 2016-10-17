AM_CXXFLAGS = $(BOOST_CPPFLAGS) $(EIGEN_CPPFLAGS) $(DYNET_CPPFLAGS) $(OPENMP_CXXFLAGS) -I$(srcdir)/..

LIBCPP = cnpy.cpp cnpy-utils.cpp
lib_LTLIBRARIES = libcnpy.la

libcnpy_la_SOURCES = ${LIBCPP}
libcnpy_la_LDFLAGS = -version-info 0:0:0


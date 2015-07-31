AM_CXXFLAGS = $(BOOST_CPPFLAGS) $(CNN_CPPFLAGS) $(EIGEN_CPPFLAGS) $(OPENMP_CXXFLAGS) -I$(srcdir)/..

noinst_PROGRAMS = test-lamtram
TESTS = test-lamtram

test_lamtram_SOURCES = \
    test-lamtram.cc \
    test-neural-lm.cc \
    test-encoder-attentional.cc \
    test-vocabulary.cc

test_lamtram_LDADD = \
	../lamtram/liblamtram.la \
    $(CNN_LDFLAGS) \
	$(BOOST_LDFLAGS) \
	$(BOOST_UNIT_TEST_FRAMEWORK_LIB) \
    $(BOOST_PROGRAM_OPTIONS_LIB) \
    $(BOOST_SERIALIZATION_LIB) \
    $(OPENMP_CXXFLAGS)

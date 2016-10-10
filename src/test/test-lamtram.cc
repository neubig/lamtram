#define BOOST_TEST_MODULE "lamtram Tests"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <dynet/init.h>

// Set up DYNET
struct CnnSetup {
    CnnSetup()   { 
        int zero = 0;
        char** null = NULL;
        dynet::initialize(zero, null);
    }
    ~CnnSetup()  { /* shutdown your allocator/check memory leaks here */ }
};

BOOST_GLOBAL_FIXTURE( CnnSetup );

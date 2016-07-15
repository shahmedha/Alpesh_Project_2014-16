#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>

#include "counters_map.h"

#define BOOST_TEST_MODULE counters_map test
#define BOOST_TEST_NO_MAIN 1
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

using namespace std;

const unsigned int N = 1000000;
const unsigned int MAX_V = 100000;

unsigned int K[N];

unsigned int random_key() {
    unsigned int k;
    do {
	k = static_cast<unsigned int>(random());
    } while (k == 0);
    return k;
}

void init_keys() {
    for(unsigned int i = 0; i < N; ++i)
	K[i] = random_key();
}

void test_hashmap_maxsize(unsigned int s, unsigned int n) {
    unsigned int i = 0;
    map<unsigned int, unsigned char> mm;
    siena_impl::counters_map m;
    
    unsigned char A;
    unsigned char B;
    for(unsigned int j = 0; j < n; ++j) {
	A = m.plus_one(K[i]);
	B = ++mm[K[i]];
	BOOST_REQUIRE_EQUAL( A, B);
	++i;
	if (i == s)
	    i = 0;
    }
}

bool init_function() {
    init_keys();

    for (unsigned int M = 10; M < 10000000; M *= 10) {
	for (unsigned int S = 10; S < 1000000; S *= 10) {
	    boost::unit_test::framework::master_test_suite().
		add( BOOST_TEST_CASE( boost::bind( &test_hashmap_maxsize, S, M ) ) );
	}
    }
    return true;
}
    
int main( int argc, char* argv[] ) {
    return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
}

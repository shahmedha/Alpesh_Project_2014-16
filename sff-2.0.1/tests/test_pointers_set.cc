#include <iostream>
#include <cstdlib>
#include <cassert>
#include <set>

#include "pointers_set.h"

#define BOOST_TEST_MODULE counters_map test
#define BOOST_TEST_NO_MAIN 1
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

using namespace std;

const unsigned int N = 1000000;
const unsigned int MAX_V = 100000;

void * K[N];

class C1 {
public:
    char c;
};

class C2 {
public:
    int i;
    int j;
};

class C3 {
public:
    int v[23];
};


void * random_key() {
    switch (static_cast<unsigned int>(random()) & 3) {
    case 0: return new char[3];
    case 1: return new C1();
    case 2: return new C2();
    default: return new C3();
    }
}

void init_keys() {
    for(unsigned int i = 0; i < N; ++i)
	K[i] = random_key();
}

void test_hashmap_maxsize(unsigned int s, unsigned int n) {
    siena_impl::pointers_set X;

    unsigned int i = 0;
    set<void *> XX;
    
    bool A, B;
    for(unsigned int j = 0; j < n; ++j) {
	A = X.insert(K[i]);
	B = XX.insert(K[i]).second;
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

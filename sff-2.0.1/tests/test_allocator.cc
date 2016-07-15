// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003 University of Colorado
//
//  Siena is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Siena is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "allocator.h"

#define BOOST_TEST_MODULE counters_map test
#define BOOST_TEST_NO_MAIN 1
#define BOOST_TEST_DYN_LINK 1

#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>

using std::cout;
using std::endl;

struct X {
    char c;
    
    X() : c('X') {};
};

struct Y {
    char c;
    int i;

    Y(int y) : c('Y'), i(y) {};
};

struct Z {
    char c;
    int i;
    void * p;

    X x;
    Y y;

    Z(int z) : c('Z'), i(z), p(this), y(z) {};
};

int test_function() {
    siena_impl::batch_allocator mem;
    X * x;
    Y * y;
    Z * z;

    for(int i = 0; i < 100000; ++i) {
        x = new (mem) X;
        y = new (mem) Y(i);
        z = new (mem) Z(i);

        if (x->c != 'X' || y->i != i || z->i != i)
            return 1;
    }
    unsigned long es = 100000 * (sizeof(X)+sizeof(Y)+sizeof(Z));
    BOOST_CHECK(mem.size() >= es);
    BOOST_CHECK(mem.allocated_size() >= mem.size());
    unsigned long prev_allocated_size = mem.allocated_size();
    mem.recycle();
    BOOST_CHECK_EQUAL(mem.size(), 0);
    BOOST_CHECK_EQUAL(mem.allocated_size(), prev_allocated_size);
    for(int i = 0; i < 100000; ++i) {
        x = new (mem) X;
        y = new (mem) Y(i);
        z = new (mem) Z(i);
        if (x->c != 'X' || y->i != i || z->i != i)
            return 1;
    }
    BOOST_CHECK(mem.size() >= es);
    BOOST_CHECK(mem.allocated_size() >= mem.size());
    mem.clear();
    BOOST_CHECK_EQUAL(mem.allocated_size(), 0);
    BOOST_CHECK_EQUAL(mem.size(), 0);
    return 0;
}

bool init_function() {

    boost::unit_test::framework::master_test_suite().
		add( BOOST_TEST_CASE( &test_function ) );
    return true;
}

int main( int argc, char* argv[] ) {
    return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
}

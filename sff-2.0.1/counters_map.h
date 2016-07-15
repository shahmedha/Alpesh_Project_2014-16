// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2012 Antonio Carzaniga
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
#ifndef COUNTERS_MAP_H
#define COUNTERS_MAP_H

#ifdef COUNTERS_MAP_USES_FTALLOCATOR
#include "allocator.h"
#endif

namespace siena_impl {

/** A fast map : id -> counter that supports only a counter increment.
 *
 *  In the initial mapping, all counters are at 0, then one may
 *  increment a counter and return the incremented value.  This map is
 *  implemented as a hash table with open addressing where the
 *  increment is computed with double hashing.
 *
 *  The table expands by a factor of two every time the load exceeds
 *  1/2 of the table size.  The table may only grow, so there is no
 *  shrinking.
 *
 *  ASSUMPTIONS: keys are uniformly distributed between an initial
 *  range over the unsigned integers.  In fact, the hash function used
 *  in this implementation is the identity function.
 */
class counters_map {
public:
    typedef unsigned char counter_t;
    typedef unsigned int key_t;

private:
    counter_t * counters; 	// points to the table of counters
    key_t * keys;		// points to the table of keys

    unsigned int load;		// number of (non-zero) counters in the table
    unsigned int size_mask;	// size == 2^k, so n % size == n & (size - 1)
    unsigned int table_size;	// allocated size; always a power of two.

    static const unsigned int INITIAL_SIZE = 1U << 10;

#ifdef COUNTERS_MAP_USES_FTALLOCATOR
    batch_allocator mem;
#endif

public:
    explicit counters_map() 
	: load(0) {
	create_table(INITIAL_SIZE);
    }

    ~counters_map() {
#ifdef COUNTERS_MAP_USES_FTALLOCATOR
	mem.clear();
#else
	delete[](counters);
	delete[](keys);
#endif
    }

    counter_t plus_one (key_t k);

private:
    void create_table(unsigned int size);
    void rehash();
};

} // end namespace siena_impl

#endif

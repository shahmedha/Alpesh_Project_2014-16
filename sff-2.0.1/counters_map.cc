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
#include "counters_map.h"
#include <cstring>

namespace siena_impl {

counters_map::counter_t counters_map::plus_one (key_t k) {
    // We assume that keys are uniformly distributed, so the hash
    // is simply the key, and the index into the table is the hash
    // modulo the size of the table
    unsigned int h = k & size_mask;
    // This is the double-hasing increment.  The double-hasing is
    // simply the complement of the hash, which is the key itself.
    // Notice that the increment must be relatively prime with the
    // size of the table.  In our case, the size is always a power
    // of two, so we simply make sure that the increment is odd by
    // setting the least significant bit.
    unsigned int step = (~h | 1) & size_mask;
    do {
	if (keys[h] == 0) {	// we look for an empty slot in the table
	    keys[h] = k;
	    counters[h] = 1;
	    ++load;
	    if (load * 2 > table_size)
		rehash();
	    return 1;
	}			// or we return the counter if we find it
	if (keys[h] == k)
	    return ++counters[h];
	// or we iterate if there is a collision
	h = (h + step) & size_mask;
    } while (true);
}

void counters_map::create_table(unsigned int size) {
    table_size = size;
#ifdef COUNTERS_MAP_USES_FTALLOCATOR
    keys = new (mem) key_t[size];
    counters = new (mem) counter_t[size];
#else
    keys = new key_t[size];
    counters = new counter_t[size];
#endif
    memset(keys, 0, size * sizeof(key_t));
    size_mask = size - 1;
}

void counters_map::rehash() {
    counter_t * old_counters = counters;
    key_t * old_keys = keys;
    unsigned int old_table_size = table_size;

    create_table(table_size << 1);
    for(unsigned int i = 0; i < old_table_size; ++i) {
	unsigned int k = old_keys[i];
	if (k != 0) {
	    // See the documentation of plus_one().
	    unsigned int h = k & size_mask;
	    unsigned int step = (~h | 1) & size_mask;

	    while (keys[h] != 0 && keys[h] != k) {
		h = (h + step) & size_mask;
	    }
	    keys[h] = k;
	    counters[h] = old_counters[i];
	}
    }
#ifndef COUNTERS_MAP_USES_FTALLOCATOR
    delete[](old_counters);
    delete[](old_keys);
#endif
}

} // end namespace siena_impl


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
#include <cstring>
#include <cstdlib>
#include <cstddef>

#include "pointers_set.h"

namespace siena_impl {

static inline size_t hash(const void * p) {
    static const size_t mask = ~(static_cast<size_t>(0));
    static const size_t alignment = offsetof(
					     struct {
						 char x; 
						 union {
						     long l;
						     void * p;
						 } y;},
					     y);

    return (reinterpret_cast<unsigned long>(p) / alignment) & mask;
}

bool pointers_set::insert (key_t k) {
    // We assume that keys are uniformly distributed, so the hash
    // is simply the key, and the index into the table is the hash
    // modulo the size of the table
    size_t h = hash(k) & size_mask;
    // This is the double-hasing increment.  The double-hasing is
    // simply the complement of the hash, which is the key itself.
    // Notice that the increment must be relatively prime with the
    // size of the table.  In our case, the size is always a power
    // of two, so we simply make sure that the increment is odd by
    // setting the least significant bit.
    size_t step = (~h | 1) & size_mask;
    do {
	if (keys[h] == 0) {	// we look for an empty slot in the table
	    keys[h] = k;
	    ++load;
	    if (load * 2 > table_size)
		rehash();
	    return true;
	}			// or we return false if the key already exists
	if (keys[h] == k)
	    return false;
	// or we iterate if there is a collision
	h = (h + step) & size_mask;
    } while (true);
}

void pointers_set::create_table(size_t size) {
    table_size = size;
    keys = new key_t[size];
    memset(keys, 0, size * sizeof(key_t));
    size_mask = size - 1;
}

void pointers_set::rehash() {
    key_t * old_keys = keys;
    size_t old_table_size = table_size;

    create_table(table_size << 1);
    for(size_t i = 0; i < old_table_size; ++i) {
	key_t k = old_keys[i];
	if (k != 0) {
	    size_t h = hash(k) & size_mask;
	    size_t step = (~h | 1) & size_mask;

	    while (keys[h] != 0 && keys[h] != k) {
		h = (h + step) & size_mask;
	    }
	    keys[h] = k;
	}
    }
    delete[](old_keys);
}

} // end namespace siena_impl


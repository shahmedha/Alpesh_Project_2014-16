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
#ifndef POINTERS_SET_H
#define POINTERS_SET_H

#include <cstddef>

namespace siena_impl {

/** A fast set : void* -> true/false that supports only insertion.
 *
 *  This set is implemented as a hash table with open addressing where
 *  the increment is computed with double hashing.  The table expands
 *  by a factor of two every time the load exceeds 1/2 of the table
 *  size.  The table may only grow, so there is no shrinking.
 */
class pointers_set {
public:
    typedef const void * key_t;

private:
    key_t * keys;		// points to the table of keys

    size_t load;		// number of (non-zero) counters in the table
    size_t size_mask;		// size == 2^k, so n % size == n & (size - 1)
    size_t table_size;		// allocated size; always a power of two.

    static const size_t INITIAL_SIZE = static_cast<size_t>(1U) << 10;

public:
    explicit pointers_set() 
	: load(0) {
	create_table(INITIAL_SIZE);
    }

    ~pointers_set() {
	delete[](keys);
    }

    bool insert (key_t k);

private:
    void create_table(size_t size);
    void rehash();
};

} // end namespace siena_impl

#endif

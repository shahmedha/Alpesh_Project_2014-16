// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2002 University of Colorado
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
#include "allocator.h"
#include "bitvector.h"

namespace siena_impl {

//
// map of a position value for the bitvector
//
//       index 2     index 1     block   element
// ...+-----------+-----------+--------+---------+
//
bool ibitvector::test(register size_t pos) const {
    if (pos > size) return false;
    register bv_block atom_mask = BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos);
    pos /= BV_BITS_PER_BLOCK;
    size_t block_pos = pos % block_size;
    pos /= index_size;
    const block * b = &first_block;
    
    if (pos) {
	const index * bi = first_block.up;
	if (bi == 0) return false;

	size_t index_addr = pos % index_size;
	unsigned char level = 0;

	while (pos /= index_size) {
	    bi = bi->up;
	    if (bi == 0) return false;
	    ++level;
	    index_addr = index_addr * index_size + (pos % index_size);
	}
	while(level-- > 0) {
	    bi = bi->down[index_addr % index_size].i;
	    if (bi == 0) return false;
	    index_addr /= index_size;
	}
	b = bi->down[index_addr % index_size].b;
	if (b == 0) return false;
    }

    return (b->elements[block_pos] & atom_mask) != 0;
}

bool ibitvector::set(register size_t pos, batch_allocator & ftmemory) {
    if (pos > size) size = pos;
    register bv_block atom_mask = BV_BLOCK_ONE << BV_POS_IN_BLOCK(pos);
    pos /= BV_BITS_PER_BLOCK;
    size_t block_pos = pos % block_size;
    pos /= index_size;

    block * b = &first_block;
    
    if (pos) {
	index * p_bi;
	index ** bi = &(first_block.up);
	size_t index_addr = 0;
	unsigned char level = 0;

	do {
	    ++level;
	    index_addr = index_addr * index_size + (pos % index_size);
	    pos /= index_size;
	    if (*bi == 0) *bi = new (ftmemory) index(0);
	    p_bi = *bi;
	    bi = &(p_bi->up);
	} while (pos);
	
	index_or_block * iob = &(p_bi->down[index_addr % index_size]);

	while (--level > 0) {
	    index_addr /= index_size;
	    if (iob->i == 0) iob->i = new (ftmemory) index(p_bi);
	    p_bi = iob->i;
	    iob = &(p_bi->down[index_addr % index_size]);
	}
	if (iob->b == 0) iob->b = new (ftmemory)block(p_bi);
	b = iob->b;
    }
    if (b->elements[block_pos] & atom_mask) {
	return true;
    } else {
	b->elements[block_pos] |= atom_mask;
	++count;
	return false;
    }
}

//
// map of a position value for the bitvector
//
//       index 2     index 1     block   element
// ...+-----------+-----------+--------+---------+
//
void bitvector::set(const ibitvector & x) {
    count += set(elements, &(elements[element_size()]), 
		 x.first_block.elements, 
		 &(x.first_block.elements[ibitvector::block_size]));

    ibitvector::iterator i(x.first_block);
    const ibitvector::block * b;
    size_t end;
    while ((b = i.next_block()) != 0) {
	if (i.element_address() >= element_size()) return;
	end = i.element_address() + ibitvector::block_size < element_size()
							     ? i.element_address() + ibitvector::block_size : element_size();
	count += set(&(elements[i.element_address()]), &(elements[end]),
		     b->elements, &(b->elements[ibitvector::block_size]));
    }
}

} // end namespace siena_impl

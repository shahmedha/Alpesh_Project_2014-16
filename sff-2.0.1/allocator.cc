// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
#include <cstdlib>		// for malloc, free, offsetof
#include <cassert>
#include <new>			// for std::bad_alloc

#include "allocator.h"

namespace siena_impl {

static inline size_t round_up_to_alignment(size_t s) {
    //
    // Here's a little bit of magic.  We write this "compile-time test" to
    // have the compiler figure out what the maximum memory alignment
    // should be on the current platform.  It's a compile-time test
    // meaning that ALIGNMENT will be optimized out at compile-time, and
    // everything will work nicely and efficiently.
    //
    static const unsigned int ALIGNMENT = offsetof(
						   struct {
						       char x; 
						       union {
							   char c;
							   int i;
							   long l;
							   bool b;
							   double d;
#ifdef ALLOCATOR_SUPPORTS_LONG_DOUBLE
							   long double ld;
#endif
							   long long ll;
							   void* vp;
							   void(*vfp)(void);
						       } y;},
						   y);

    size_t extra = s % ALIGNMENT;
    return  (extra == 0) ? s : (s - extra + ALIGNMENT);
}

void batch_allocator::attach_malloc_block (void *ptr) {
    large_block *lb = new (*this) large_block;
    lb->ptr = ptr;
    lb->next = largeblist;
    largeblist = lb;
}

void * batch_allocator::allocate(size_t s) {
    if (s > BLOCK_SIZE) {
	void *ptr = malloc(s);
	if (ptr) {
	    try {
		attach_malloc_block (ptr);
	    } catch (std::bad_alloc) {
		free(ptr);
		throw;
	    }
	} else {
	    throw std::bad_alloc();
	}
	return ptr;
    }

    s = round_up_to_alignment(s);
    if (s + free_pos > BLOCK_SIZE || blist == 0) {
	block * nb;
	if (freeblist) {
	    nb = freeblist;
	    freeblist = nb->next;
	} else {
	    nb =  static_cast<block*>(malloc(sizeof(block)));
	    if (!nb) 
		throw std::bad_alloc();
	    ++bcount;
	}
	nb->next = blist;
	blist = nb;
	free_pos = 0;
    }
    void  * res = &(blist->bytes[free_pos]);
    free_pos += s;
    normal_size += s;
    return res;
}

void batch_allocator::recycle() {
    batch_allocator *ptr = suballocs;
    while (ptr) {
	// Not a typo.  Suballocators are often allocated within the master
	// allocator, so we cannot recycle memory in the suballocator.
	ptr->clear ();
	ptr = ptr->next;
    }
    while (largeblist) {
	free(largeblist->ptr);
	largeblist = largeblist->next;
    }

    block ** curs = & freeblist;
    while (*curs)
	curs = & ((*curs)->next);
    *curs = blist;
    blist = 0;
    free_pos = 0;
    normal_size = 0;
    large_size = 0;
}

void batch_allocator::clear() {
    while (suballocs) {
	suballocs->clear ();
	suballocs->detach ();
    }
    while (largeblist) {
	free(largeblist->ptr);
	largeblist = largeblist->next;
    }
    block * tmp;
    while (blist) {
	tmp = blist;
	blist = blist->next;
	free(tmp);
    }
    while (freeblist) {
	tmp = freeblist;
	freeblist = freeblist->next;
	free(tmp);
    }
    free_pos = 0;
    bcount = 0;
    assert (suballocs == 0);
    assert (blist == 0);
    assert (freeblist == 0);
    assert (largeblist == 0);
    normal_size = 0;
    large_size = 0;
}

void batch_allocator::detach_sub_allocator (batch_allocator &mem) {
    batch_allocator **ptr;
    for (ptr = &suballocs; *ptr != &mem; ptr = &((*ptr)->next))
	;
    *ptr = mem.next;
    mem.master = mem.next = 0;
}

void batch_allocator::attach_sub_allocator (batch_allocator &mem) {
    mem.next = suballocs;
    mem.master = this;
    suballocs = &mem;
}

size_t batch_allocator::size() const {
    return normal_size + large_size;
}

size_t batch_allocator::allocated_size() const {
    return bcount * BLOCK_SIZE + large_size;
}

} // end namespace siena_impl

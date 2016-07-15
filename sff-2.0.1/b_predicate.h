// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
//  Copyright (C) 2005 Antonio Carzaniga
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
#ifndef B_PREDICATE_H
#define B_PREDICATE_H

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <siena/attributes.h>
#include <siena/forwarding.h>
#include "allocator.h"

#include "bloom_filter.h"
#include "attributes_encoding.h"

namespace siena_impl {

template<class BFilterT>
BFilterT * encode_predicate(const siena::Predicate & p, batch_allocator & mem) {
    BFilterT * flist = 0; 
    siena::Predicate::Iterator * pi = p.first();
    if (pi) {
	bloom_filter<> b;
	do {
	    b.clear();
	    encode(b, pi);

	    bool must_add = true;
	    for(BFilterT * f = flist; f != 0; f = f->next) {
		if (b.covers(f->b)) {
		    //
		    // NOTICE: this "covers" relation is defined
		    // in the sense of the Bloom filter coverage,
		    // i.e., b1 covers b2 if all the 1-bit
		    // positions of b2 are also 1-bit positions of
		    // b1.  In other words, if ((b1 & b2) == b2).
		    // Because of the semantics of the Bloom
		    // filter encoding, this relation is exactly
		    // the opposite of the usual covering
		    // relations.  This is why, we do not add new
		    // filters that cover previous filters (in the
		    // sense of the Bloom filter), and instead we
		    // swap in the new filter whenever that *is
		    // covered* by a previous filter.
		    //
		    must_add = false;
		    break;
		} else if (f->b.covers(b)) {
		    f->b = b;
		    //
		    // not only we replace the old one with the
		    // new one here, but we must also remove all
		    // other previous filters that are also
		    // covered (in the sense of the Bloom filter)
		    // by the new one.  So, this is what we do
		    // with this loop.
		    //
		    while (f->next) {
			if (f->next->b.covers(b)) {
			    f->next = f->next->next;
			} else {
			    f = f->next;
			}
		    }
		    must_add = false;
		    break;
		}
	    }
	    if (must_add) {
		flist = new (mem)BFilterT(flist, b);
	    }
	} while(pi->next());
	delete(pi);
    }
    return flist;
}
} // end namespace siena_impl

#endif

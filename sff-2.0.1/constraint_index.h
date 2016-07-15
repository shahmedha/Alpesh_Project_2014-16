// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2004 University of Colorado
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
#ifndef CONSTRAINT_INDEX_H
#define CONSTRAINT_INDEX_H

#include "allocator.h"
#include <siena/forwarding.h>
#include <siena/fwdtable.h>

#include "v_index.h"

namespace siena_impl {

/** generic index of <em>equals</em>, <em>less-than</em>, and
 *  <em>greater-than</em> constraints.
 *
 *  This template index works for <em>equals</em>, <em>less-than</em>,
 *  and <em>greater-than</em> constraints for all the types that
 *  implement operators <code>==</code>, <code>&lt;</code>, and
 *  <code>&gt;</code>.  In particular, this template is optimal for
 *  numeric types (integers or floating point).
 **/
template<class T>
class constraint_index {
private:
    typedef v_index<T> v_index_t;

    v_index_t lt_map;
    v_index_t gt_map;
    v_index_t ne_map;
    v_index_t eq_map;

    fwd_constraint * any_value;

public:
    constraint_index(): any_value(0) {};

    fwd_constraint * add_any(batch_allocator & ftmemory) {
	if (any_value == 0) 
	    any_value = new (ftmemory)fwd_constraint();
	return any_value;
    }

    fwd_constraint * add_ne(const T & v, batch_allocator & ftmemory) {
	return ne_map.add(v, ftmemory);
    }

    fwd_constraint * add_lt(const T & v, batch_allocator & ftmemory) {
	return lt_map.add_r(v, ftmemory);
    }

    fwd_constraint * add_gt(const T & v, batch_allocator & ftmemory) {
	return gt_map.add(v, ftmemory);
    }

    fwd_constraint * add_eq(const T & v, batch_allocator & ftmemory) {
	return eq_map.add(v, ftmemory);
    }

    void consolidate(batch_allocator & mem) {
	lt_map.consolidate(mem);
	gt_map.consolidate(mem);
	ne_map.consolidate(mem);
	eq_map.consolidate(mem);
    }

    bool match(const T & v, c_processor & p) const {
	//
	// any value first
	//
	if (any_value) 
	    if (p.process_constraint(any_value))
		return true;

	typename v_index_t::iterator i;
	//
	// equality first
	//
	i = eq_map.find(v);
	if (i != eq_map.end()) 
	    if (p.process_constraint(i->c))
		return true;
	//
	// now inequality
	//
	for(i = ne_map.begin(); i != ne_map.end(); ++i) {
	    if (i->v < v) {		// match the ne constraint 
					// for all less-than values
		if (p.process_constraint(i->c))
		    return true;
	    } else {			// then skip the ne constraint 
		if (i->v == v) {	// with  == value, if one exists
		    if (++i == ne_map.end())
			break;
		}
		do { 		// then process every other ne constraint
		    if (p.process_constraint(i->c))
			return true;
		} while (++i != ne_map.end());
		break;
	    }
	}
	//
	// now greater-than
	//
	for(i = gt_map.begin(); i != gt_map.end() && i->v < v; ++i) 
	    if (p.process_constraint(i->c))
		return true;
	//
	// now less-than
	//
	for(i = lt_map.begin(); i != lt_map.end() && i->v > v; ++i)
	    if (p.process_constraint(i->c))
		return true;
	return false;
    }
};

} // end namespace siena_impl

#endif

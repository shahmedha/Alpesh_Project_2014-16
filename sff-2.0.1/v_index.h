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
#ifndef V_INDEX_H
#define V_INDEX_H

#include "allocator.h"

namespace siena_impl {

class fwd_constraint;

/** generic sorted, searchable map of constraints.
 *
 *  Conceptually, v_index is a sorted map from T values to
 *  constraints, and in fact its access methods are a subset of those
 *  of the std::map template.  The main idea behind the design of
 *  v_index is to provide an extremely fast and memory-efficient
 *  find() after the data structure has been "consolidated."
 *  Specifically, values can be inserted into a v_index though the
 *  add() method.  After all values are in, the v_index can be
 *  compacted using the consolidate() method, at which point v_index
 *  is ready to be used as a read-only sorted set through the find()
 *  method.
 *
 *  The requirement on the template parameter T is essentially the
 *  same as std::set.  In particular, T must implement the usual
 *  equality and inequality operators (==, !=, &lt;, and &gt;).
 *
 *  The implementation of v_index is extremely simple: the initial
 *  data structure is a linked list, with in-order, O(n) insertion.
 *  consolidate() transforms the list in a fixed-size vector where
 *  find() is implemented with a binary search.
 **/
template<class T>
class v_index {
public:
    struct node {
	T v;
	fwd_constraint * c;
	
	node(): v(), c(0) {}
	node(const T & val, fwd_constraint * con) 
	    : v(val), c(con) {}
	
	node & operator = (const node & x) { 
	    v = x.v; 
	    c = x.c; 
	    return *this; 
	}
    };

    struct v_link : public node {
	v_link * next;

	v_link(const T & v, fwd_constraint * d, v_link * n) 
	    : node(v,d), next(n) {}
    };

private:
    v_link * head;
    unsigned int size;
    node * values;

public:
    v_index() : head(0), size(0), values(0) {};

    /** adds a (constraint) value for the natural ordering of values
     */
    fwd_constraint * add(const T & v, batch_allocator & ftmem) {
	v_link ** curs = & head;
	while((*curs) && (*curs)->v < v) {
	    curs = &((*curs)->next);
	}
	if (!(*curs) || (*curs)->v != v) {
	    *curs = new v_link(v, new (ftmem)fwd_constraint(), *curs);
	    ++size;
	}
	return (*curs)->c;
    }

    /** adds a (constraint) value for the <em>reverse</em> natural
     *  ordering of values
     */
    fwd_constraint * add_r(const T & v, batch_allocator & ftmem) {
	v_link ** curs = &head;
	while(*curs && (*curs)->v > v) {
	    curs = &((*curs)->next);
	}
	if (!(*curs) || (*curs)->v != v) {
	    *curs = new v_link(v, new (ftmem)fwd_constraint(), *curs);
	    ++size;
	}
	return (*curs)->c;
    }

    typedef const node * iterator;

    const iterator begin() const { return & values[0]; }
    const iterator end() const { return & values[size]; }

    const iterator find(const T & v) const {
	register unsigned int first = 0;
	register unsigned int last = size;
	register unsigned int middle;
	while (first < last) {
	    middle = (last + first) >> 1; // (first + len) / 2
	    if (values[middle].v < v) {
		first = middle + 1;
	    } else if (values[middle].v > v) {
		last = middle;
	    } else {
		return & values[middle];
	    } 
	}
	return end();
    }

    void consolidate(batch_allocator & ftmem) {
	if (head) {
	    v_link * tmp;
	    values = new (ftmem) node[size];
	    int i = 0;
	    do {
		values[i] = *head;
		++i;
		tmp = head;
		head = head->next;
		delete(tmp);
	    } while (head);
	}
    }
};

} // end namespace siena_impl

#endif

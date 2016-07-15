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
#ifndef STRING_INDEX_H
#define STRING_INDEX_H

#include <siena/attributes.h>
#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/fwdtable.h>

namespace siena_impl {

/** index of string constraints.
 *
 *  This data structure supports <em>equals</em>, <em>less-than</em>,
 *  <em>greater-than</em>, <em>prefix</em>, <em>suffix</em>, and
 *  <em>substring</em> constraints on strings
 **/
class string_index {
public:

    string_index() : root(0), any_value(0) {}

    void		consolidate(batch_allocator &) {};
    fwd_constraint *	add_eq(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_ne(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_lt(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_gt(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_pf(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_sf(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_ss(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_re(const siena::String & v, batch_allocator &);
    fwd_constraint *	add_any(batch_allocator &);

    bool		match(const siena::String & v, c_processor & p) const;

private:
    typedef unsigned char	mask_t;

    /** descriptor of a complete match in a <code>string_index</code> **/
    struct complete_descr {
	fwd_constraint *		eq;
	fwd_constraint *		ne;
	fwd_constraint *		lt;
	fwd_constraint *		gt;
	fwd_constraint *		sf;
	complete_descr *	next_lt;
	complete_descr *	next_gt;
    };

    /** descriptor of a partial match in a <code>string_index</code> **/
    struct partial_descr {
	fwd_constraint *		pf;
	fwd_constraint *		ss;
    };

    struct node {
	union {
	    complete_descr *	c_data;
	    partial_descr *	p_data;
	};
	int	c;
	node *	up;
	node *	left;
	node *	middle;
	node *	right;
	mask_t	mask;

	node(int xc, node * xup, mask_t m) 
	    : c_data(0), c(xc), up(xup), left(0), middle(0), right(0), mask(m) {}
    };

    node *	insert_complete(const char *, const char *, 
				batch_allocator &, const mask_t);
    node *	insert_partial(const char * s, const char * end,
			       batch_allocator & a, const mask_t);
    bool	tst_match(const siena::String & v, c_processor & p) const;

    node * root;
    fwd_constraint * any_value;


    static node * backtrack_next(node * p, const mask_t mask);
    static void insert_between(complete_descr * d, node * p, node * n);
    static node * next(node * p, const mask_t mask);
    static node * prev(node * p, const mask_t mask);
    static const node * last(const node * pp, const mask_t mask);
    static const node * upper_bound(const node * p, 
				    const char * begin, const char * end,
				    const mask_t mask);
};

} // end namespace siena_impl

#endif

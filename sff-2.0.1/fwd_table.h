// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2005 University of Colorado
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
#ifndef FWD_TABLE_H
#define FWD_TABLE_H

#include <siena/forwarding.h>
#include "allocator.h"

#include "bitvector.h"
#ifndef WITH_STATIC_COUNTERS
#include "counters_map.h"
#endif
#include "bloom_filter.h"

namespace siena_impl {

class fwd_filter;

/* link in a single-link list for filters */
class f_list {
public:
    /* the filter */
    const fwd_filter * f;
    /* link to the next filter in the list */
    const f_list * next;

    /* constructor */
    f_list(fwd_filter * xf, const f_list * xn) : f(xf), next(xn) {};
};

/*  constraint descriptor
 *
 *  it is essentially a set of pointers to the filters in which this
 *  constraint appears.
 */
class fwd_constraint {
public:
    /** @brief list of filters associated with this constraint
     *
     *  All the filters in which this constraint appears.
     *
     *  <p>Right now, we add filters to this list as they are passed
     *  to the forwarding table.  It might be possible and
     *  advantageous to figure out an optimal ordering of this list in
     *  order to speed up the maching process.  For example, we could
     *  sort the filters based on their size.
     **/
    const fwd_filter * f;
    /** link to the next filter in the list **/
    const f_list * next;

    /** @brief constructor **/
    fwd_constraint() : f(0), next(0) {};
};

/*  this constraint processor represents the main "iterator" of the
 *  matching function.
 */
class c_processor {

public:
    /** initializes the constraint processor */
    c_processor(unsigned int ifcount, siena::MatchHandler & p, bitvector * v, 
		const siena::Message & n) 
	: if_mask(v), processor(p), target(ifcount) {
#ifdef WITH_STATIC_COUNTERS
	static unsigned int msg_id_factory = 0;
	msg_id = ++msg_id_factory;
#endif
	siena::Message::Iterator * i  = n.first(); 
	if (i != 0) {
	    do {
		a_set.add(i->name().begin, i->name().end);
	    } while (i->next());
	    delete(i);
	}
    }

    /** method called by the matching function for every matched
     *  constraint.
     *
     *  This method implements a fundamental part of the matching
     *  function.
     */
    bool process_constraint(const fwd_constraint *);

private:
    /** set of interfaces that can be ignored in the matching process.
     *
     *  An interface can be ignored by the matching process if it has
     *  already been matched and processed or if it was excluded by
     *  the pre-processing function.
     */
    bitvector *			if_mask;
    /** output processor
     */
    siena::MatchHandler &	processor;
    /** total number of interfaces we can match.
     *
     *  we maintain this value so that we can stop the matching
     *  process immediately once we have matched all interfaces.
     */
    const unsigned int		target;

#ifdef WITH_STATIC_COUNTERS
    /** current message id
     */
    unsigned int		msg_id;
#else
    /** table of counters for partially-matched filters.
     *
     *  Each filter considered in the matching process has an
     *  associated counter that records the number of constraints
     *  matched so far for that filter.
     */
    counters_map fmap;
#endif
    /** Bloom filter representing the set of the attribute names in
     *  the message being evaluated.
     *
     *  This Bloom filter provides a negative check against filters.
     *  In essence, a message must contain all the attributes in a
     *  filter, otherwise that filter has no chance of being matched.
     */
    bloom_filter<64,4>		a_set;
};

} // end namespace siena_impl

#endif

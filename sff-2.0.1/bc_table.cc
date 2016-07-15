// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003 University of Colorado
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <siena/attributes.h>
#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/bctable.h>

#include "b_table.h"
#include "bitvector.h"
#include "counters_map.h"
#include "hash.h"
#include "bloom_filter.h"
#include "attributes_encoding.h"

#include "timers.h"

namespace siena_impl {

struct bc_interface {
    const siena::InterfaceId interface;
    const ifid_t id;

    /** builds an interface descriptor with the given identifiers **/
    bc_interface(siena::InterfaceId xif, ifid_t xid) : interface(xif), id(xid) {};
};

struct bc_filter {
    bc_filter(bc_interface * xi, unsigned int xid) 
	: i(xi), size(0), id(xid) {};

    bc_interface * i;
    unsigned int size;
    /*  filter identifier used as a key in the table of counters in
     *  the matching algorithm
     */
    unsigned int id;
};

struct bc_filterList {
    bc_filterList * next;
    bc_filter * f;

    bc_filterList(bc_filterList * nxt, bc_filter * fi) : next(nxt), f(fi) {};
};

class bc_table : public b_table {
public:
    bc_table();
    virtual ~bc_table();

    virtual void consolidate();
    virtual void match(const siena::Message &, siena::MatchHandler &) const;

private:
    /** @brief list of predicates.
     *
     *  each link in this linked list is a pair <predicate,interface-id>
     **/
    bc_filterList *	positions[CONFIG_BLOOM_FILTER_SIZE];
    unsigned int	f_counter;
    unsigned int	if_counter;
};

bc_table::bc_table() : f_counter(0), if_counter(0) {
    for(unsigned int i = 0; i < CONFIG_BLOOM_FILTER_SIZE; ++i)
	positions[i] = 0;
}

bc_table::~bc_table() { 
    memory.clear();
}

void bc_table::consolidate() { 

    TIMER_PUSH(consolidate_timer);

    for(const b_predicate * p = plist; p; p = p->next) {
	bc_interface * I = new (memory) bc_interface(p->id, if_counter++);

	for(const b_filter * f = p->flist; f; f = f->next) {
	    bc_filter * bcf = new (memory)bc_filter(I, ++f_counter);

	    for(unsigned int i = 0; i < CONFIG_BLOOM_FILTER_SIZE; ++i) {
		if (f->b.test(i)) {
		    positions[i] = new (memory)bc_filterList(positions[i], bcf);
		    ++(bcf->size);
		}
	    }
	}
    }
    TIMER_POP();
}

class bc_processor {

public:
    bc_processor(unsigned int if_count, 
		 siena::MatchHandler & h, 
		 unsigned int max_pos,
		 bc_filterList * const * parray)
	: fmap(), 
	  pos_mask(max_pos),
	  if_mask(if_count), 
	  handler(h), 
	  target(if_count),
	  positions(parray),
	  WIDTH(max_pos),
	  complete(false)
    { };

    bool set(unsigned int pos);

    bool matching_complete() { return complete; }

private:
    /** table of counters for partially-matched filters.
     *
     *  Each filter considered in the matching process has an
     *  associated counter that records the number of constraints
     *  matched so far for that filter.
     */
    counters_map		fmap;

    bitvector 			pos_mask;

    /** set of interfaces that can be ignored in the matching process.
     *
     *  An interface can be ignored by the matching process if it has
     *  already been matched and processed or if it was excluded by
     *  the pre-processing function.
     */
    bitvector 			if_mask;
    /** output processor
     */
    siena::MatchHandler &	handler;
    /** total number of interfaces we can match.
     *
     *  we maintain this value so that we can stop the matching
     *  process immediately once we have matched all interfaces.
     */
    unsigned int	target;

    bc_filterList * const * positions;

public:
    const unsigned int WIDTH;

private:
    bool		complete;
};

bool bc_processor::set(unsigned int pos) {
    if (pos_mask.set(pos)) 
	return false;

    for(const bc_filterList * l = positions[pos]; l; l = l->next) {
	// for each filter where the pos-th position is 1
	//
	if (! if_mask.test(l->f->i->id)) {
	    // if the interface to which the filter is associated has
	    // not been matched already
	    //
	    if (fmap.plus_one(l->f->id) >= l->f->size) {
		// if the filter has been matched, then mark this
		// interface as "matched", and call the processor on
		// it.  interface id into the result.
		//
		if_mask.set(l->f->i->id); 

		TIMER_PUSH(forward_timer);

		bool output_result = handler.output(l->f->i->interface);

		TIMER_POP();

		if (output_result) 
		    return complete = true;

		if (--target == 0) 
		    return complete = true;
	    }
	}
    }
    return false;
}

void bc_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    siena::Message::Iterator * a  = m.first();
    if (a) {
	TIMER_PUSH(match_timer);

	bc_processor p(if_counter, h, CONFIG_BLOOM_FILTER_SIZE, positions);
	bloom_filter_wrapper< CONFIG_BLOOM_FILTER_SIZE, CONFIG_BLOOM_FILTER_K, 
			      bc_processor > bf(p);
	do {
	    encode_attribute(bf, a);
	} while (!p.matching_complete() && a->next());
	delete(a);

	TIMER_POP();
    }
}

} // end namespace siena_impl

siena::BTable * siena::BCTable::create() {
    return new siena_impl::bc_table();
}


// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003 University of Colorado
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string>

#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/btable.h>

#include "b_table.h"
#include "b_predicate.h"
#include "attributes_encoding.h"

#include "timers.h"

namespace siena_impl {

/** @brief inteface identifier within the matching algorithm.  
 *
 *  As opposed to <code>if_t</code> which identifies user-specified
 *  interface numbers, this is going to be used for the identification
 *  of interfaces within the matching algorithm, which may require a
 *  set of contiguous identifiers.  So, for example, the user may
 *  specify interfaces 6, 78, and 200, while the internal
 *  identification would be 0, 1, and 2 respectively (or similar).  I
 *  treat it as a different type (from <code>if_t</code>) because I
 *  don't want to mix them up (now that I write this, I'm not even
 *  sure the compiler would complain. Oh well.)
 **/
typedef unsigned int		ifid_t;

/** @brief implementation of the forwarding table based on Bloom
 *         filters.
 *
 *  This implementation extends the \link siena_impl::b_table b_table
 *  algorithm\endlink simply by sorting the Bloom filters within each
 *  predicate.  Bloom filters are sorted by their Hamming weight in
 *  ascending order.  The rationale for this is that the basic b_table
 *  algorithm shortcuts the evaluation (only) when a match is found.
 *  Therefore, it is advantageous to try "smaller" Bloom filters
 *  first, because those are more likely to yield a match.
 **/
class sorted_b_table : public b_table {
public:
    sorted_b_table();

    virtual void consolidate();
};

b_table::b_table() : plist(0) {}
b_table::~b_table() {}
sorted_b_table::sorted_b_table() : b_table() {}

void b_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {
    TIMER_PUSH(ifconfig_timer);
    b_filter * flist = encode_predicate<b_filter>(p, memory);
    if (flist) {
	plist = new (memory)b_predicate(plist, id, flist);
    }
    TIMER_POP();
}

void b_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    b_predicate * p = plist;
    if (!p) return;


    TIMER_PUSH(bloom_encoding_timer);

    bloom_filter<> b;
    encode(b, &m);
    
    TIMER_POP();

    TIMER_PUSH(match_timer);

    do {
	for(b_filter * f = p->flist; f != 0; f = f->next) {
	    if (b.covers(f->b)) {

		TIMER_PUSH(forward_timer);

		bool output_result = h.output(p->id);

		TIMER_POP();

		if (output_result) {
		    TIMER_POP();
		    return;
		}
		break;
	    }
	}
    } while ((p = p->next));

    TIMER_POP();
}

void b_table::clear() {
    memory.clear();
    plist = 0;
}
void b_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t b_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t b_table::bytesize() const {
    return memory.size();
}

static void sort_flist(struct b_predicate * p) {
    // this is plain and simple insertion sort.  We build a new
    // (sorted) list of filters starting with new_head
    b_filter * head = p->flist;
    b_filter * new_head = 0;

    while (head) {
	b_filter ** curr = &new_head; 
	while ((*curr) && head->b.count() > (*curr)->b.count())
	    curr = &((*curr)->next);
	b_filter * tmp = head->next;
	head->next = *curr;
	*curr = head;
	head = tmp;
    }
    p->flist = new_head;
}

void sorted_b_table::consolidate() {

    TIMER_PUSH(consolidate_timer);

    for(b_predicate * p = plist; (p); p = p->next)
	sort_flist(p);

    TIMER_POP();
}

} // end namespace siena_impl

siena::BTable * siena::BTable::create() {
    return new siena_impl::b_table();
}

siena::BTable * siena::SortedBTable::create() {
    return new siena_impl::sorted_b_table();
}


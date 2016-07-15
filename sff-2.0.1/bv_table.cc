// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
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

#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/bvtable.h>

#include "b_predicate.h"
#include "bitvector.h"
#include "bloom_filter.h"
#include "attributes_encoding.h"

#include "timers.h"

#include "b_table.h"

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

#ifdef DEBUG_OUTPUT
#define DBG(x) {std::cout << x;}
#else
#define DBG(x) 
#endif

#ifdef DEBUG_OUTPUT
std::ostream & operator << (std::ostream & os, const bitvector & b) {
    for(unsigned int i = 0; i < b.get_size(); ++i) 
	os << (b[i] ? '1' : '0');
    return os;
}
#endif

/** @brief a predicate in a BVTable.
 *
 *  This is an extremely simple data structure.  The predicate is
 *  represented as a linked list of filters.  Then, each predicate
 *  refers directly to its interface id, and is itself a link in a
 *  linked list of predicates, which is what makes up the BVTable.
 *
 *  The consolidation of the data structure builds up a matrix of
 *  bitvectors masks.  Each line i in the matrix corresponds to the
 *  i-th 8-bit block in the encoded message.  Specifically, for each
 *  8-bit value v, the i-th row contains a bitvector representing the
 *  set of filters in this predicate that would _not_ match if value v
 *  was found at position i.  
 **/
struct bv_predicate {
    bv_predicate * next;
    ifid_t id;
    bitvector * masks[bloom_filter<>::B8Size][256];

    bv_predicate(bv_predicate *n, ifid_t i) : next(n), id(i) {}

    bool matches(const bloom_filter<> & b) const;
};

static inline bool covers(unsigned char x, unsigned char y) {
    return ((x & y) == y);
} 

class bv_table : public siena::BVTable {
public:
    bv_table();

    virtual ~bv_table();

    virtual void ifconfig(siena::InterfaceId, const siena::Predicate &);
    virtual void consolidate();

    virtual void match(const siena::Message &, siena::MatchHandler &) const;

    virtual void clear();
    virtual void clear_recycle();

    virtual size_t allocated_bytesize() const;
    virtual size_t bytesize() const;

protected:
    /** @brief Protected allocator of the forwarding table.  
     *
     *  All the data structure forming the forwarding table are
     *  allocated through this memory management system.
     **/
    batch_allocator		memory;

    /** @brief list of predicates.
     *
     *  each link in this linked list is a pair <predicate,interface-id>
     **/
    bv_predicate *	plist;
};

bv_table::bv_table() : plist(0) {}
bv_table::~bv_table() {}

void bv_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {

    TIMER_PUSH(ifconfig_timer);

    batch_allocator tmp_mem;
    b_filter * flist = encode_predicate<b_filter>(p, tmp_mem);
    if (!flist) {

	TIMER_POP();

	return;
    }
    plist = new (memory)bv_predicate(plist, id);

    unsigned int flist_size = 0;
    for(b_filter * curs = flist; (curs); curs = curs->next)
	++flist_size;

    for(unsigned int pos = 0; pos < bloom_filter<>::B8Size; ++pos) {
	for(unsigned int v = 0; v < 256; ++v) {
	    FABitvector * b = new (memory)FABitvector(memory, 
						      flist_size, true);
	    unsigned int fpos = 0;
	    for(b_filter * curs = flist; (curs); curs = curs->next) {
		if (!covers(v, curs->b.bv8(pos))) 
		    b->clear(fpos);
		++fpos;
	    }
	    plist->masks[pos][v] = b; // TO BE CHANGED WITH A PATRICIA TRIE DB 
	}
    }
    TIMER_POP();
}

// the matching function for a predicate represented as an XDD is
// trivial: starting at the root of the XDD, we simply look at the
// current block in the input set 
//
bool bv_predicate::matches(const bloom_filter<> & b) const {
    bitvector res(*masks[0][b.bv8(0)]);
    if (res.get_count() == 0) {
	DBG("#" << std::endl);
	return false;
    } 

    unsigned pos = 1; 
    do {
	DBG('[' << pos << "] " << res << " / " 
	    << (*masks[pos][b.bv8(pos)]) << std::endl);
	res &= *masks[pos][b.bv8(pos)];
	if (res.get_count() == 0) {
	    DBG("#" << std::endl);
	    return false;
	}
	++pos;
    } while (pos < bloom_filter<>::B8Size);
    DBG("!" << std::endl);
    return true;
}

void bv_table::consolidate() { }

void bv_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    DBG("match: " << std::endl);

    TIMER_PUSH(bloom_encoding_timer);

    bloom_filter<> b;
    encode(b, &m);
    
    DBG("> " <<  b << std::endl);

    TIMER_POP();

    TIMER_PUSH(match_timer);

    for (bv_predicate * p = plist; (p); p = p->next) {
	if (p->matches(b)) {

	    TIMER_PUSH(forward_timer);

	    bool output_result = h.output(p->id);

	    TIMER_POP();

	    if (output_result) {
		TIMER_POP();
		return;
	    }
	}
    }
    TIMER_POP();
}

void bv_table::clear() {
    memory.clear();
    plist = 0;
}
void bv_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t bv_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t bv_table::bytesize() const {
    return memory.size();
}

} // end namespace siena_impl

siena::BVTable * siena::BVTable::create() {
    return new siena_impl::bv_table();
}



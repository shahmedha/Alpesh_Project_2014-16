// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2013 Antonio Carzaniga
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

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <siena/forwarding.h>
#include <siena/tagstable.h>
#include <siena/ttable.h>

#include "allocator.h"
#include "bloom_filter.h"
#include "attributes_encoding.h"
#include "timers.h"

/** 
 *  This file contains the implementation of a very basic tags table.
 *  See the t_table class below.
 **/
namespace siena_impl {

/** @brief inteface identifier within the matching algorithm.  
 **/
typedef unsigned int ifid_t;

/** @brief a tag in a tagset: a link in a linked list of tags.
 **/
struct t_tag {
    const char * const begin;
    const char * const end;
    t_tag * next;

    static const char * copy_string(const std::string &v, batch_allocator & mem) {
	char * res = new (mem) char[v.size()];
	memcpy(res, v.data(), v.size());
	return res;
    }

    t_tag(batch_allocator & mem, const std::string &v, t_tag * n) 
	: begin(copy_string(v,mem)), end(begin + v.size()), next(n) {}

    size_t length() const {
	return end - begin;
    }
};

static bool operator < (const t_tag & a, const t_tag & b) {
    size_t a_len = a.length();
    size_t b_len = b.length();
    if (a_len < b_len) {
	return (memcmp(a.begin, b.begin, a_len) <= 0);
    } else { // a_len >= b_len
	return (memcmp(a.begin, b.begin, b_len) < 0);
    }
}

typedef std::vector<std::string> str_vec;

/** @brief a tagset in a predicate: a link in a linked list of tagsets.
 **/
struct t_tagset {
    t_tag * first_tag;
    t_tagset * next_tagset;

    t_tagset(t_tagset * n): first_tag(0), next_tagset(n) {};

    void add_tag(const std::string & v, batch_allocator & mem);
    bool is_subset_of(str_vec::const_iterator begin,
		      str_vec::const_iterator end) const;
};
/** @brief adds a tag to a tagset
 *
 *  inserts tags in a tagset in sorted lexicographical order.
 **/
void t_tagset::add_tag(const std::string & v, batch_allocator & mem) {
    //
    // essentially this is the inner loop of an insertion sort: we
    // assume that the list of tags that starts with 'tagset' is
    // sorted in non-decreasing order, and then we insert the new one
    // initially at the beginning of the list:
    first_tag = new (mem) t_tag(mem, v, first_tag);

    // then we start from that element and we swap it with the next
    // one if the next is smaller, until we reach the end of the list
    // or a next element that is not smaller than the new one.
    t_tag ** tp = &first_tag;

    while((*tp)->next != 0 && *((*tp)->next) < **tp) {
	t_tag * tmp = *tp;
	*tp = tmp->next;
	tmp->next = (*tp)->next;
	(*tp)->next = tmp;
	tp = &((*tp)->next);
    }
}

bool t_tagset::is_subset_of(str_vec::const_iterator begin,
			    str_vec::const_iterator end) const {
    for(const t_tag * t = first_tag; t != 0; t = t->next) {
	begin = std::find(begin, end, std::string(t->begin, t->end));
	if (begin == end)
	    return false;
    }
    return true;
}

struct t_if_descr {
    ifid_t id;
    t_tagset * first_tagset;
    t_if_descr *next;

    t_if_descr(t_if_descr *n, ifid_t i) 
	: id(i), first_tagset(0), next(n) {}
};

/** @brief a very simple implementation of a TagsFIB.
 *
 *  Basically, this is a table in which a tagset list is associated
 *  with each interface and in which insertion and matching are
 *  sequential algorithms.  In particular, matching goes through each
 *  tagset of each tagset to find subsets of the given tagset.
 **/
class t_table : public siena::TTable {
public:
    t_table();
    virtual ~t_table();

    virtual void ifconfig(siena::InterfaceId, const siena::TagSetList &);
    virtual void match(const siena::TagSet &, siena::MatchHandler &) const;

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
    batch_allocator	memory;

    /** @brief list of predicates.
     *
     *  each link in this linked list is a pair <predicate,interface-id>
     **/
    t_if_descr *	plist;
};

t_table::t_table() : plist(0) {}
t_table::~t_table() {}

void t_table::ifconfig(siena::InterfaceId id, const siena::TagSetList & tl) {

    TIMER_PUSH(ifconfig_timer);

    siena::TagSetList::Iterator * tli = tl.first();

    if (tli) {
	plist = new (memory)t_if_descr(plist, id);

	do {
	    siena::TagSet::Iterator * tsi = tli->first();

	    if (tsi) {
		plist->first_tagset = new (memory)t_tagset(plist->first_tagset);
		do {
		    plist->first_tagset->add_tag(tsi->to_string(), memory);
		} while (tsi->next());
		delete(tsi);
	    }
	} while(tli->next());
	delete(tli);
    }

    TIMER_POP();
}

void t_table::match(const siena::TagSet & ts, siena::MatchHandler & h) const {

    t_if_descr * p = plist;
    if (!p) return;

    TIMER_PUSH(match_timer);

    siena::TagSet::Iterator * tsi = ts.first();

    if (tsi) {

	str_vec m;
	do {
	    m.push_back(tsi->to_string());
	} while (tsi->next());
	delete(tsi);
	
	std::sort(m.begin(), m.end());

	do {
	    for(const t_tagset * s = p->first_tagset; s != 0; s = s->next_tagset) {
		if (s->is_subset_of(m.begin(), m.end())) {

		    TIMER_PUSH(forward_timer);

		    bool output_result = h.output(p->id);

		    TIMER_POP();

		    if (output_result) {
			TIMER_POP();
			return;
		    }
		}
	    }
	} while ((p = p->next));
    }
    TIMER_POP();
}

void t_table::clear() {
    memory.clear();
    plist = 0;
}
void t_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t t_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t t_table::bytesize() const {
    return memory.size();
}


} // end namespace siena_impl

siena::TTable * siena::TTable::create() {
    return new siena_impl::t_table();
}


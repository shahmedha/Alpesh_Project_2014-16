// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
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

// none of this needs to be compiled if we don't have CUDD.
//
#ifdef HAVE_CUDD
#include <cstdio> // needed by cudd
#include <cudd.h>
#include <cuddInt.h>

#ifdef DEBUG_OUTPUT
#include <iostream>
#define DBG(x) {std::cout << x;}
#else
#define DBG(x)
#endif
#include <new>
#include <string>
#include <map>

#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/bddbtable.h>

#include "bset_encoding.h"
#include "b_table.h"

#include "bdd_table.h"

#include "timers.h"

namespace siena_impl {

/** @brief implementation of the forwarding table based on Bloom
 *         filters. This implementation consolidates each predicate
 *         into a ZDD, which is then used for matching.
 *
 *  @see bdd_table_base
 **/
class zdd_table : public siena::ZDDBTable {
public:
    zdd_table();
    virtual ~zdd_table();

    virtual void ifconfig(siena::InterfaceId, const siena::Predicate &);
    virtual void match(const siena::Message &, siena::MatchHandler &) const;

    virtual void clear();
    virtual void clear_recycle();

    virtual void consolidate();

    virtual size_t allocated_bytesize() const;
    virtual size_t bytesize() const;

private:
    /** @brief Protected allocator of the forwarding table.  
     *
     *  All the data structure forming the forwarding table are
     *  allocated through this memory management system.
     **/
    batch_allocator memory;

    class if_descr;

    if_descr * zddlist;
};

class zdd_table::if_descr {
    // item in a list of interface descriptors for a ZDD BTable
public:
    if_descr * next;
    const siena::InterfaceId id;

public:
    if_descr(if_descr *nx, siena::InterfaceId i)
	: next(nx), id(i), true_node(0), zdd(0) {};
    ~if_descr();

    void compile_dd(const DdNode *, DdManager *, batch_allocator &);
    bool match(const bloom_filter<> & b, siena::MatchHandler & h) const;
    static bool match_list(const if_descr * l, 
			   const bloom_filter<> & b, siena::MatchHandler & h);

private:
    bdd_node * true_node;
    bdd_node * zdd;

    cb_queue * link_node(const cb_queue * Q, bool dir, 
			 nodemap_t & nodemap, batch_allocator & mem);
};

bool zdd_table::if_descr::match_list(const if_descr * n,
				     const bloom_filter<> & b, 
				     siena::MatchHandler & h) {
    while(n) { 
	if (n->match(b, h)) 
	    return true;
	n = n->next;
    }
    return false;
}

static DdNode * False_Node = new DdNode();
static DdNode * True_Node = 0;

static const DdNode * skip_irrelevant_nodes(const DdNode * n) {
    // skips all nodes of the form  (n)
    //                              | |
    //                              v v
    //                              (x)
    // to the first non-irrelevant node
    for(;;) {
	if (Cudd_IsConstant((Cudd_Regular(n)))) {
	    return (Cudd_V(n) == 0) ? False_Node : True_Node;
	} else if (Cudd_T(n) == Cudd_E(n)) {
	    n = Cudd_T(n);
	} else {
	    return n;
	}
    }
}

cb_queue * zdd_table::if_descr::link_node(const cb_queue * Q, 
					  bool direction,
					  nodemap_t & nodemap, 
					  batch_allocator & mem) {
    cb_queue * new_q = 0;
    bdd_node * n = Q->n;
    bdd_node ** link = (direction) ? &(n->true_link) : &(n->false_link);
    const DdNode * dest = skip_irrelevant_nodes((direction) ? 
						Cudd_T(Q->cudd_node) 
						: Cudd_E(Q->cudd_node));
    if (dest == True_Node) {
	DBG("n" << n << " -> TRUE");
	if (!true_node) {
	    true_node = new (mem) bdd_node();
	}
	*link = true_node;
    } else if (dest == False_Node) {
	DBG("n" << n << " -> FALSE");
	*link = 0;
    } else {
	std::pair<nodemap_t::iterator,bool> pib;
	pib = nodemap.insert(nodemap_t::value_type(dest, 0));
	if (pib.second) {
	    *link = new (mem) bdd_node((Cudd_Regular(dest))->index);
	    DBG("n" << *link << " [label=\"" << (*link)->pos << "\"];" << std::endl);
	    (*pib.first).second = *link;
	    new_q = new cb_queue(0, dest, (*pib.first).second);
	} else {
	    *link = (*pib.first).second;
	}
	DBG("n" << n << " -> n" << *link);
    }
    DBG(" [label=\"" << (direction ? '1' : '0') << "\"];" << std::endl);
    return new_q;
}

void zdd_table::if_descr::compile_dd(const DdNode * cudd, 
				     DdManager * mgr, 
				     batch_allocator & mem) {
    if (!cudd) return;
    nodemap_t nodemap;
    cb_queue * Q = 0;
    cb_queue * lastQ = 0;

    DBG("digraph ZDD" << id << " {" << std::endl);
    cudd = skip_irrelevant_nodes(cudd);
    if (cudd == True_Node) {
	zdd = true_node = new (mem) bdd_node();
    } else if(cudd == False_Node) {
	zdd = 0;
    } else {
	zdd = new (mem) bdd_node((Cudd_Regular(cudd))->index);
	DBG("n" << zdd << " [label=\"" << zdd->pos << "\"];" << std::endl);
	Q = lastQ = new cb_queue(0, cudd, zdd);
    }
    while(Q) { 
	cb_queue * tmp;

	if ((tmp = link_node(Q, true, nodemap, mem))) {
	    lastQ = lastQ->next = tmp;
	}
	if ((tmp = link_node(Q, false, nodemap, mem))) {
	    lastQ = lastQ->next = tmp;
	}
	tmp = Q;
	Q = Q->next;
	delete(tmp);
    }
    DBG("}" << std::endl);
}

bool zdd_table::if_descr::match(const bloom_filter<> & b, siena::MatchHandler & h) const {
    const bdd_node * n = zdd;
    while(n) {
	if (n == true_node) {
	    return false;
	}
	if (b[n->pos]) {
	    n = n->false_link;
	} else {
	    n = n->true_link;
	}
    }
    return h.output(id);
}

zdd_table::zdd_table() : zddlist(0) {}

zdd_table::~zdd_table() {
    clear();
}

static DdNode * bset_to_zdd(const bset_t & B, DdManager * mgr) {
    //
    // INPUT: a set B over the universe U = {1, 2, ..., BSize - 1}
    //
    // OUTPUT: a bdd f representing the set of all sets over U that do
    //         not contain one or more elements of B
    //
    DdNode * f = 0;
    for(unsigned int i = 0; i < BSize; ++i) {
	if(B[i]) {
	    // s0 is the set of all sets that do not contain element i
	    DdNode * s0 = Cudd_zddIthVar(mgr, i);
	    if (!s0) {
		goto error_cleanup;
	    }
	    Cudd_Ref(s0);
	    if (!f) {
		f = s0;
	    } else {
		DdNode * U = Cudd_zddUnion(mgr, f, s0);
		if (!U) {
		    goto error_cleanup;
		}
		Cudd_Ref(U);
		Cudd_RecursiveDerefZdd(mgr, f);
		Cudd_RecursiveDerefZdd(mgr, s0);
		f = U;
	    }
	}
    }
    return (f) ? f : False_Node;

 error_cleanup:
    Cudd_RecursiveDerefZdd(mgr, f);
    return 0;
}

void zdd_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {
    siena::Predicate::Iterator * pi = p.first();
    if (!pi) return;

    TIMER_PUSH(ifconfig_timer);

    DBG("ifconfig " << id);
    zddlist = new (memory)if_descr(zddlist, id);

    DdManager * mgr = Cudd_Init(0, BSize, 
				CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    Cudd_AutodynEnableZdd(mgr, CUDD_REORDER_SYMM_SIFT);
    DdNode * zdd = 0;

    do {
	DBG('.' << std::flush);
	siena::Filter::Iterator * fi = pi->first();
	if (fi) {
	    bloom_filter<> b;

	    do {
		encode_constraint(b, fi); 
	    } while (fi->next());
	    delete(fi);

	    DdNode * f_zdd = bset_to_zdd(b, mgr);
	    if (!f_zdd) {
		goto error_cleanup;
	    } else if (f_zdd != False_Node) {
		// if there are existing conjunctions, add this one
		if (zdd) {
		    DdNode * U = Cudd_zddIntersect(mgr, f_zdd, zdd);
		    if (U) {
			Cudd_Ref(U);
			Cudd_RecursiveDerefZdd(mgr, zdd);
			zdd = U;
		    } else {
			goto error_cleanup;
		    }
		    Cudd_RecursiveDerefZdd(mgr, f_zdd);
		} else {
		    zdd = f_zdd;
		}
	    }
	}
    } while(pi->next());
    delete(pi);
    DBG("reducing ZDD..." << std::flush)
	Cudd_zddReduceHeap(mgr, CUDD_REORDER_SAME, 0);
    DBG(std::endl)
	zddlist->compile_dd(zdd, mgr, memory);
    Cudd_Quit(mgr);

    TIMER_POP();

    return;

 error_cleanup:
    DBG("unable to construct ZDD." << std::endl);
    Cudd_Quit(mgr);
    delete(pi);

    TIMER_POP();

    throw std::bad_alloc();
}

void zdd_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    if (!zddlist) return;

    siena::Message::Iterator * i  = m.first();
    if (!i) return;

    TIMER_PUSH(bloom_encoding_timer);

    bloom_filter<> b;
    do {
	encode_attribute(b, i); 
    } while (i->next());
    delete(i);

    TIMER_POP();

    TIMER_PUSH(match_timer);

    if_descr::match_list(zddlist, b, h);

    TIMER_POP();
}

void zdd_table::clear() {
    // NOTE: don't have to delete the ZDDInterfaces because they are
    // allocated using the allocator and are reclaimed when that is
    // cleared by BTable::clear() and BTable::clear_recycle().
    zddlist = 0;
    memory.clear();
}

void zdd_table::clear_recycle() {
    // NOTE: don't have to delete the ZDDInterfaces because they are
    // allocated using the allocator and are reclaimed when we call
    // memory.clear()
    zddlist = 0;
    memory.recycle();
}

void zdd_table::consolidate() {}

size_t zdd_table::allocated_bytesize() const {
    unsigned int res = memory.allocated_size();
    return res;
}

size_t zdd_table::bytesize() const {
    unsigned int res = memory.size();
    return res;
}


}; // end namespace siena_impl

siena::ZDDBTable * siena::ZDDBTable::create() {
    return new siena_impl::zdd_table();
}

#endif // HAVE_CUDD

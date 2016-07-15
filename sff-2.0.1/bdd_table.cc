// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
//  Copyright (C) 2004-2005 Antonio Carzaniga
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
#include <exception>

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
 *         into a BDD, which is then used for matching.
 *
 *  @see bdd_table_base
 **/
class bdd_table : public siena::BDDBTable {
public:
    bdd_table();
    virtual ~bdd_table();

    virtual void ifconfig(siena::InterfaceId, const siena::Predicate &);
    virtual void match(const siena::Message &, siena::MatchHandler &) const;

    virtual void clear();
    virtual void clear_recycle();

    virtual void consolidate();

    virtual size_t allocated_bytesize() const;
    virtual size_t bytesize() const;

private:
    batch_allocator memory;
    class if_descr;

    if_descr * bddlist;
};

class bdd_table::if_descr {
public:
    if_descr * next;
    const siena::InterfaceId id;

public:
    if_descr(if_descr *nx, siena::InterfaceId i)
	: next(nx), id(i), true_node(0), bdd(0) {};
    ~if_descr();

    void compile_dd(const DdNode *, DdManager *, batch_allocator &);
    bool match(const bloom_filter<> & b, siena::MatchHandler & h) const;
    static bool match_list(const if_descr * l, 
			   const bloom_filter<> & b, siena::MatchHandler & h);

private:
    bdd_node * true_node;
    bdd_node * bdd;

    cb_queue * link_node(const cb_queue * Q, bool direction, const DdNode * one,
			 nodemap_t & nodemap, batch_allocator & mem);
};

bool bdd_table::if_descr::match_list(const if_descr * n,
				     const bloom_filter<> & b, 
				     siena::MatchHandler & h) {
    while(n) { 
	if (n->match(b, h)) 
	    return true;
	n = n->next;
    }
    return false;
}

bool bdd_table::if_descr::match(const bloom_filter<> & b, 
				siena::MatchHandler & h) const {
    const bdd_node * n = bdd;
    while(n) {
	if (n == true_node) {

	    TIMER_PUSH(forward_timer);

	    bool output_result = h.output(id);

	    TIMER_POP();

	    return output_result;
	}
	if (b[n->pos]) {
	    n = n->true_link;
	} else {
	    n = n->false_link;
	}
    }
    return false;
}

cb_queue * bdd_table::if_descr::link_node(const cb_queue * Q, 
					  bool direction,
					  const DdNode * one,
					  nodemap_t & nodemap, 
					  batch_allocator & mem) {
    cb_queue * new_q = 0;
    bdd_node * n = Q->n;
    bdd_node ** link = (direction) ? &(n->true_link) : &(n->false_link);
    const DdNode * dest = (direction) ? Cudd_T(Q->cudd_node) : Cudd_E(Q->cudd_node);

    if (Cudd_IsConstant(dest)) {
	if (dest == one) {
	    DBG("n" << n << " -> TRUE");
	    if (!true_node) {
		true_node = new (mem) bdd_node();
	    }
	    *link = true_node;
	} else {
	    DBG("n" << n << " -> FALSE");
	    *link = 0;
	}
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

void bdd_table::if_descr::compile_dd(const DdNode * cudd, 
				     DdManager * mgr, 
				     batch_allocator & mem) {
    if (!cudd) return;
    nodemap_t nodemap; 		// Map: DdNode * --> bdd_node *
    cb_queue * Q = 0;		// Q and lastQ are the head and tail,
    cb_queue * lastQ = 0;	// respectively, of a queue of nodes
				// used in a breadth-first visit of
				// the BDD
    const DdNode * one = Cudd_ReadOne(mgr);
    DBG("digraph BDD" << id << " {" << std::endl);

    if (Cudd_IsConstant(cudd)) {
	if (cudd == one) {
	    bdd = true_node = new (mem) bdd_node();
	} else {
	    bdd = 0;
	}
    } else {
	bdd = new (mem) bdd_node((Cudd_Regular(cudd))->index);
	DBG("n" << bdd << " [label=\"" << bdd->pos << "\"];" << std::endl);	Q = lastQ = new cb_queue(0, cudd, bdd);
    }
    while(Q) {
	cb_queue * tmp;

	if ((tmp = link_node(Q, true, one, nodemap, mem))) {
	    lastQ = lastQ->next = tmp;
	}
	if ((tmp = link_node(Q, false, one, nodemap, mem))) {
	    lastQ = lastQ->next = tmp;
	}
	tmp = Q;
	Q = Q->next;
	delete(tmp);
    }
    DBG("}" << std::endl);
}

bdd_table::bdd_table() : bddlist(0) {}

bdd_table::~bdd_table() {
    clear();
}

void bdd_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {
    siena::Predicate::Iterator * pi = p.first();
    if (!pi) return;

    DBG("ifconfig " << id << std::endl);

    TIMER_PUSH(ifconfig_timer);

    bddlist = new (memory)if_descr(bddlist, id);

    DdManager * mgr = Cudd_Init(BSize, 0, 
				CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    Cudd_AutodynEnable(mgr, CUDD_REORDER_SYMM_SIFT);
    DdNode * bdd = 0;

    do {
	siena::Filter::Iterator * fi = pi->first();
	if (fi) {
	    bloom_filter<> b;

	    do {
		encode_constraint(b, fi); 
	    } while (fi->next());
	    delete(fi);

	    DdNode * f = 0;
	    for(unsigned int i = 0; i < BSize; ++i) {
		if (b[i]) {
		    DdNode * var = Cudd_bddIthVar(mgr,i);
		    if (!var) { goto error_cleanup; }
		    Cudd_Ref(var);
		    if (!f) {
			f = var;
		    } else {
			DdNode * tmp;
			tmp = Cudd_bddAnd(mgr, var, f);
			if (!tmp) { goto error_cleanup; }
			Cudd_Ref(tmp);
			Cudd_RecursiveDeref(mgr, f);
			Cudd_RecursiveDeref(mgr, var);
			f = tmp;
		    }
		}
	    }
	    if (f) {
		if (!bdd) {
		    bdd = f;
		} else {
		    DdNode * tmp = Cudd_bddOr(mgr, bdd, f);
		    if (!tmp) { goto error_cleanup; }
		    Cudd_Ref(tmp);
		    Cudd_RecursiveDeref(mgr, f);
		    Cudd_RecursiveDeref(mgr, bdd);
		    bdd = tmp;
		}
	    }
	}
	DBG("      \rNodes: " << Cudd_DagSize(bdd) << std::flush);
    } while(pi->next());
    delete(pi);
    DBG(std::endl);

    Cudd_ReduceHeap(mgr, CUDD_REORDER_SAME, 0);
    bddlist->compile_dd(bdd, mgr, memory);
    Cudd_Quit(mgr);

    TIMER_POP();

    return;

 error_cleanup:
#ifdef DEBUG_OUTPUT
    std::cout << "unable to construct BDD." << std::endl;
#endif
    delete(pi);
    Cudd_Quit(mgr);

    TIMER_POP();

    // perhaps I should define a specific exception for this case...
    throw std::bad_alloc();
}

void bdd_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    if (!bddlist) return;

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
    if_descr::match_list(bddlist, b, h);
    TIMER_POP();
}

void bdd_table::clear() {
    // NOTE: don't have to delete the BDDInterfaces because they are
    // allocated using the allocator and are reclaimed when that is
    // cleared by BTable::clear() and BTable::clear_recycle().
    bddlist = 0;
    memory.clear();
}

void bdd_table::clear_recycle() {
    // NOTE: don't have to delete the BDDInterfaces because they are
    // allocated using the allocator and are reclaimed when we call
    // memory.clear()
    bddlist = 0;
    memory.recycle();
}

void bdd_table::consolidate() {}

size_t bdd_table::allocated_bytesize() const {
    unsigned int res = memory.allocated_size();
    return res;
}

size_t bdd_table::bytesize() const {
    unsigned int res = memory.size();
    return res;
}

}; // end namespace siena_impl

siena::BDDBTable * siena::BDDBTable::create() {
    return new siena_impl::bdd_table();
}

#endif // HAVE_CUDD

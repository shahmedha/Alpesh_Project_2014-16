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
#include <set>
#include <vector>

#include <siena/forwarding.h>
#include <siena/tagstable.h>
#include <siena/btrietable.h>

#include "allocator.h"
#include "bloom_filter.h"
#include "attributes_encoding.h"
#include "timers.h"

/** 
 *  This file contains the implementation of b_trie_table.
 *
 *  b_trie_table is a forwarding table based on Bloom filters, where
 *  each interface is associated with a predicate consisting of a set
 *  of Bloom filters, each representing a conjunction of constraints,
 *  or in some cases "tags".  Thus the b_trie_table is essentially a
 *  list of b_trie_predicate objects, each representing the association
 *  of a predicate to an interface.  
 * 
 *  A Bloom filter, representing a filter, is here represented as a
 *  set of its constituent 1-bits, that is a set of the positions of
 *  its bits set to 1.  So, for example, 00100010110 would be
 *  represented as the set {2,6,8,9} (first position is 0).
 * 
 *  Each predicate, which again consists of a set of Bloom filters, is
 *  then represented as a binary trie.  Each node in the trie
 *  represents an element in a Bloom filter, that is, the position of
 *  a 1-bit.  The elements are in ascending order in the trie, so a
 *  sub-trie rooted at node T with value T.pos represents all the
 *  Bloom filters that contain all the elements between the root and
 *  T, which are less than T.pos, and that may contain other elements
 *  greater than T.pos.  More specifically, a node T has two children
 *  T.t and T.f pointing to the subtrees that represet the Bloom
 *  filters that share the prefix of T (from the root to T) and,
 *  respectively, the Bloom filters that contain T.pos and those that
 *  do not contain T.pos.
 *
 *  The central algorithm of b_trie_table is a subset test, implemented
 *  in suffix_contains_subset.  This test takes a set F of positions
 *  and a binary trie T, and checks whether one of the filters
 *  represented in T is a subset of F.  This algorithm is used in
 *  building the trie representing a predicate in order to avoid
 *  inserting redundant filters (i.e., filters that cover another
 *  filter alredy in the trie).  The same algorithm is also used in the
 *  matching algorithm, since that is essentially a subset search.
 *
 **/

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

/** @brief representation of a Bloom filters as a set of positions.
 **/
class pos_set_filter {
public:
    typedef unsigned int pos_t;
    typedef std::set<pos_t>::iterator iterator;
    typedef std::set<pos_t>::const_iterator const_iterator;

    static const unsigned int WIDTH = CONFIG_BLOOM_FILTER_SIZE;

private:
    std::set<pos_t> elements;

public:
    pos_set_filter * next;

    void set(pos_t pos) {
	elements.insert(pos);
    }

    const_iterator begin() const {
	return elements.begin();
    }

    iterator begin() {
	return elements.begin();
    }

    const_iterator end() const {
	return elements.end();
    }

    iterator end() {
	return elements.end();
    }

    int count() const {
	return elements.size();
    }

    pos_set_filter(): elements() { }

    pos_set_filter(const std::string & s) throw(int);
    pos_set_filter & operator=(const std::string & s) throw(int);
};

class btrie_node {
public:
    pos_set_filter::pos_t pos;
    btrie_node * t;
    btrie_node * f;

    btrie_node(pos_set_filter::pos_t p): pos(p), t(0), f(0) {};
};

static bool suffix_contains_subset(bool prefix_is_good, 
				   const btrie_node * n, 
				   pos_set_filter::const_iterator fi, 
				   pos_set_filter::const_iterator end) {
    // 
    // two alternative implementations of precisely the same
    // algorithm.  The second one is the one that illustrates the
    // algorithm more clearly, and is written in completely recursive
    // form.  The first one, which is the one we actually use, simply
    // transforms the tail-recursive calls into simple iterations.
    //
#ifndef TAIL_RECURSIVE_IS_ITERATIVE
#define TAIL_RECURSIVE_IS_ITERATIVE
#endif

#ifdef TAIL_RECURSIVE_IS_ITERATIVE
    while(n != 0) {
	if (fi == end) {
	    return false;
	} else if (n->pos > *fi) {
	    ++fi; 
	} else if (n->pos < *fi) {
	    prefix_is_good = false;
	    n = n->f;
	} else if (suffix_contains_subset(false, n->f, ++fi, end)) {
	    return true;
	} else {
	    prefix_is_good = true;
	    n = n->t;
	}
    }
    return prefix_is_good;
#else
    if (n == 0)
 	return prefix_is_good;
    if (fi == end)
 	return false;
    if (n->pos > *fi) 
 	return suffix_contains_subset(prefix_is_good, n, ++fi, end);
    if (n->pos < *fi)
 	return suffix_contains_subset(false, n->f, fi, end);
    // n->pos == *fi
    ++fi;
    return suffix_contains_subset(false, n->f, fi, end)
	|| suffix_contains_subset(true, n->t, fi, end);
#endif
}

/** @brief a predicate in a b_trie_table.
 *
 *  This is an extremely simple data structure.  The predicate is
 *  represented as a linked list of filters.  Thus, b_trie_predicate is
 *  a link in such a list and refers directly to its interface id and
 *  the trie reprenting the set of filters that make up the predicate.
 **/
struct b_trie_predicate {
    btrie_node * root;
    ifid_t id;
    b_trie_predicate * next;

public:
    b_trie_predicate(b_trie_predicate *n, ifid_t i) 
	: root(0), id(i), next(n) {}

    ~b_trie_predicate(); 

    bool contains_subset(pos_set_filter::const_iterator begin, 
			 pos_set_filter::const_iterator end) const {
	return suffix_contains_subset(false, root, begin, end);
    }

    void add_filter(pos_set_filter::const_iterator begin, 
		    pos_set_filter::const_iterator end,
		    batch_allocator & memory);
};

/** @brief implementation of the forwarding table based on Bloom
 *         filters.
 *
 *  This implementation is based on an encoding of messages and
 *  filters that transforms a message \em m in a set \em Sm and a
 *  filter \em f in set \em Sf such that if \em m matches \em f then
 *  \em Sm contains \em Sf.  Sets are then represented with Bloom
 *  filters that admit a compact representation and a fast matching
 *  operation.  The matching algorithm is then based on a simple
 *  linear scan of the encoded filters.
 *
 *  Because of the nature of both the encoding of messages
 *  and filters into sets, and the representation of sets with Bloom
 *  filters, this implementation can not provide an exact match.  In
 *  particular, this implementation produces false positives, which means
 *  that a message might be forwarded to more interfaces than the ones
 *  it actually matches.  However, this implementation does not
 *  produce false negatives.  This means that a message will always go
 *  to all the interfaces whose predicate match the message.
 *
 *  b_trie_table is based on a representation of a set of Bloom filters
 *  as a trie.  More details are available in the technical
 *  documentation within the source file.
 **/
class b_trie_table : public siena::BTrieTable {
public:
    b_trie_table();
    virtual ~b_trie_table();

    virtual void ifconfig(siena::InterfaceId, const siena::Predicate &);
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
    b_trie_predicate *	plist;
};

void b_trie_predicate::add_filter(pos_set_filter::const_iterator fi, 
				pos_set_filter::const_iterator end,
				batch_allocator & memory) {
    btrie_node ** np = &root;

    while(fi != end) {
	if (*np == 0) {
	    *np = new (memory)btrie_node(*fi);
	    np = &((*np)->t);
	    ++fi;
	} else if ((*np)->pos < *fi) {
	    np = &((*np)->f);
	} else if ((*np)->pos > *fi) {
	    btrie_node * tmp = *np;
	    *np = new (memory)btrie_node(*fi);
	    (*np)->f = tmp;
	    np = &((*np)->t);
	    ++fi;
	} else { // (*np)->pos == *fi
	    np = &((*np)->t);
	    ++fi;
	}
    }
}

b_trie_table::b_trie_table() : plist(0) {}
b_trie_table::~b_trie_table() {}

void b_trie_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {

    TIMER_PUSH(ifconfig_timer);

    siena::Predicate::Iterator * pi = p.first();

    if (pi) {
	b_trie_predicate * p = new (memory)b_trie_predicate(plist, id);
	plist = p;

	pos_set_filter * fv[pos_set_filter::WIDTH];
	memset(fv,0,sizeof(fv));
	do {
	    pos_set_filter * b = new pos_set_filter();
	    bloom_filter_wrapper<CONFIG_BLOOM_FILTER_SIZE, 
				 CONFIG_BLOOM_FILTER_K, 
				 pos_set_filter> bf(*b);

	    TIMER_PUSH(bloom_encoding_timer);

	    encode(bf, pi);

	    TIMER_POP();

	    b->next = fv[b->count()];
	    fv[b->count()] = b;
	} while(pi->next());
	delete(pi);

	for (unsigned int i = 0; i < pos_set_filter::WIDTH; ++i) {
	    pos_set_filter * b = fv[i]; 
	    pos_set_filter * tmp;
	    while (b != 0) {
		if (!p->contains_subset(b->begin(), b->end()))
		    p->add_filter(b->begin(), b->end(), memory);
		tmp = b;
		b = b->next;
		delete(tmp);
	    }
	}
    }
    TIMER_POP();
}

void b_trie_table::match(const siena::Message & m, siena::MatchHandler & h) const {

    b_trie_predicate * p = plist;
    if (!p) return;

    TIMER_PUSH(bloom_encoding_timer);

    pos_set_filter b;
    bloom_filter_wrapper<CONFIG_BLOOM_FILTER_SIZE, 
			 CONFIG_BLOOM_FILTER_K, 
			 pos_set_filter> bf(b);

    encode(bf, &m);

    TIMER_POP();

    TIMER_PUSH(match_timer);

    do {
	if (p->contains_subset(b.begin(), b.end())) {

	    TIMER_PUSH(forward_timer);

	    bool output_result = h.output(p->id);

	    TIMER_POP();

	    if (output_result) {
		TIMER_POP();
		return;
	    }
	}
    } while ((p = p->next));

    TIMER_POP();
}

void b_trie_table::clear() {
    memory.clear();
    plist = 0;
}
void b_trie_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t b_trie_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t b_trie_table::bytesize() const {
    return memory.size();
}

class t_trie_table : public siena::TagsTable {
public:
    t_trie_table() : plist(0) {}
    virtual ~t_trie_table() {}

    /** @brief Associates a predicate to an interface.
     *
     *  This is the method that constructs the forwarding table.  This
     *  method must be called \em once for each interface, after
     *  the forwarding table is constructed or after it has been
     *  cleared.  Using this method twice on the same interface
     *  without clearing the forwarding table has undefined effects.
     *
     *  @see consolidate()
     **/
    virtual void ifconfig(siena::InterfaceId, const siena::TagSetList &);

    /** @brief Processes a message, calling the output() function on
     *	the given MatchHandler object for each matching interface.
     *
     *  Matches a message against the predicates stored in the
     *  forwarding table.  The result is processed through the
     *  MatchHandler passed as a parameter to this function.
     *
     *  Notice that the forwarding table must be consolidated by
     *  calling \link siena::ForwardingTable::consolidate()\endlink
     *  before this function is called.
     *
     *  @see consolidate()
     **/
    virtual void match(const siena::TagSet &, siena::MatchHandler &) const;

    /** @brief Clears the forwarding table.
     *
     *  This method removes all the associations from the forwarding
     *  table and releases allocated memory.  After a call to this
     *  method, the forwarding table is ready to be configured with
     *  \link ifconfig()\endlink.
     *
     *  @see ifconfig()
     *  @see consolidate()
     **/
    virtual void clear();

    /** @brief Clears the forwarding table.
     *
     *  This method removes all the associations from the forwarding
     *  table recycling the allocated memory. After a call to this
     *  method, the forwarding table is ready to be configured with
     *  \link ifconfig()\endlink.
     *
     *  @see ifconfig()
     *  @see consolidate()
     **/
    virtual void clear_recycle();

    /** @brief Memory allocated by the forwarding table.
     *
     *  returns the number of bytes of memory allocated by the
     *  forwarding table.  This value is always greater than or equal
     *  to the value returned by bytesize().
     **/
    virtual size_t allocated_bytesize() const;

    /** @brief Memory used by the forwarding table.
     *
     *  returns the number of bytes of memory used by the forwarding
     *  table.  This value is always less than or equal to the value
     *  returned by allocated_bytesize().
     **/
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
    b_trie_predicate *	plist;
};

void t_trie_table::ifconfig(siena::InterfaceId id, const siena::TagSetList & tl) {

    TIMER_PUSH(ifconfig_timer);

    siena::TagSetList::Iterator * tli = tl.first();

    if (tli) {
	b_trie_predicate * p = new (memory)b_trie_predicate(plist, id);
	plist = p;

	// we use the fv array to perform a bucket-sort on the hamming
	// weight of each encoded tag set.
	pos_set_filter * fv[pos_set_filter::WIDTH];
	memset(fv,0,sizeof(fv));
	do {
	    siena::TagSet::Iterator * tsi = tli->first();

	    if (tsi) {
		pos_set_filter * b = new pos_set_filter();
		bloom_filter_wrapper<CONFIG_BLOOM_FILTER_SIZE, 
				     CONFIG_BLOOM_FILTER_K, 
				     pos_set_filter> bf(*b);
		std::string v;
		do {
		    tsi->to_string(v);
		    bf.add(v.data(), v.data() + v.size());
		} while (tsi->next());
		delete(tsi);
		b->next = fv[b->count()];
		fv[b->count()] = b;
	    }
	} while(tli->next());
	delete(tli);

	for (unsigned int i = 0; i < pos_set_filter::WIDTH; ++i) {
	    pos_set_filter * b = fv[i]; 
	    pos_set_filter * tmp;
	    while (b != 0) {
		if (!p->contains_subset(b->begin(), b->end()))
		    p->add_filter(b->begin(), b->end(), memory);
		tmp = b;
		b = b->next;
		delete(tmp);
	    }
	}
    }
    TIMER_POP();
}

void t_trie_table::match(const siena::TagSet & ts, siena::MatchHandler & h) const {

    b_trie_predicate * p = plist;
    if (!p) return;

    siena::TagSet::Iterator * tsi = ts.first();

    pos_set_filter b;

    if (tsi) {
	bloom_filter_wrapper<CONFIG_BLOOM_FILTER_SIZE, 
			     CONFIG_BLOOM_FILTER_K, 
			     pos_set_filter> bf(b);
	std::string v;
	do {
	    tsi->to_string(v);
	    bf.add(v.data(), v.data() + v.size());
	} while (tsi->next());
	delete(tsi);
    }

    TIMER_PUSH(match_timer);

    do {
	if (p->contains_subset(b.begin(), b.end())) {

	    TIMER_PUSH(forward_timer);

	    bool output_result = h.output(p->id);

	    TIMER_POP();

	    if (output_result) {
		TIMER_POP();
		return;
	    }
	}
    } while ((p = p->next));

    TIMER_POP();
}

void t_trie_table::clear() {
    memory.clear();
    plist = 0;
}
void t_trie_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t t_trie_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t t_trie_table::bytesize() const {
    return memory.size();
}


} // end namespace siena_impl

siena::BTrieTable * siena::BTrieTable::create() {
    return new siena_impl::b_trie_table();
}

siena::TagsTable * siena::TagsTable::create() {
    return new siena_impl::t_trie_table();
}

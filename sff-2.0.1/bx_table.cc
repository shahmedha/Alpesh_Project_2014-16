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

#include <cstdio>
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#include <siena/forwarding.h>
#include "allocator.h"
#include <siena/bxtable.h>

#include "bloom_filter.h"
#include "attributes_encoding.h"
#include "b_predicate.h"
#include "bitvector.h"

#include "timers.h"

namespace siena_impl {

#ifdef DEBUG_OUTPUT
#define DBG(x) {std::cout << x;}
#else
#define DBG(x) 
#endif

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

/** @brief a filter in a BTable.
 *
 *  a filter is simply a link in a linked list that contains a
 *  Bloom filter, which represents the conjunction of constraints.
 **/
struct bx_filter {
    const unsigned int fpos;
    bx_filter * next;
    mutable bx_filter * next_tmp;
    bloom_filter<> b;
    
    bx_filter(bx_filter * n, const bloom_filter<> & x) 
	: fpos(((n) ? n->fpos + 1 : 0)), next(n), b(x) {};
};

struct xdd_node {
    const unsigned int pos;	// position of the input block
    xdd_node * table[256];

    xdd_node(unsigned int p): pos(p) {}
};

/** @brief a predicate in a BXTable.
 *
 *  This is an extremely simple data structure.  The predicate is
 *  represented as a linked list of filters.  Then, each predicate
 *  refers directly to its interface id, and is itself a link in a
 *  linked list of predicates, which is what makes up the BXTable.
 *
 *  The consolidation of the data structure builds up an XDD for
 *  each predicate.  See bx_table.cc for more comments on that.
 **/
struct bx_predicate {
    bx_predicate * next;
    ifid_t id;
    bx_filter * flist;
    xdd_node * xdd;
    
    bx_predicate(bx_predicate *n, ifid_t i) : next(n), id(i), 
					      flist(0), xdd(0) {}
    
    void add_filter(const siena::Filter * f, batch_allocator & mem);
    
    bool matches(const bloom_filter<> & b) const;

    void build_xdd(batch_allocator & mem);
};

static inline bool covers(unsigned char x, unsigned char y) {
    return ((x & y) == y);
} 

static inline bool intersect(unsigned char x, unsigned char y) {
    return (x & y);
} 

class bx_table : public siena::BXTable {
public:
    bx_table();

    virtual ~bx_table();

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
    bx_predicate *	plist;
};

bx_table::bx_table() : plist(0) {}
bx_table::~bx_table() { }

void bx_table::ifconfig(siena::InterfaceId id, const siena::Predicate & p) {

    TIMER_PUSH(ifconfig_timer);

    batch_allocator tmp_mem;
    bx_filter * flist = encode_predicate<bx_filter>(p, tmp_mem);;
    if (flist) {
	plist = new (memory) bx_predicate(plist, id);
	plist->flist = flist;
	plist->build_xdd(memory);
    }

    TIMER_POP();
}

//
// The problem we're trying to solve is the follwing: we want to
// represent a "databaase" set D of sets F_1, F_2, ..., F_N, where
// each set F_i is a subset of a universe U of M elements.  So:
//
// U = { x_1, x_2, ..., x_M }
// D = { F_1, F_2, ..., F_N }
// F_1 \subseteq U, F_2 \subseteq U, ..., F_N \subseteq U
//
// Given a "query" set Q \subseteq U, we want to decide if the
// database D contains a set F included in Q.  Formally:
//
// given Q \subseteq U, result is "yes" iff
// \exists F_i \in D : Q \supseteq F_i
//
// It is easy to think of the database and quesy sets as bit vectors
// of size M.  For example, a set F is represented by a bit vector B
// having the j-th bit set to 1 iff the j-th element of U belongs to
// F.  So, for example:
//
// D = { [0010110001], [0000111000], [0100011000] }
//
// does Q = [1100110010] cover any set in D?
//
// The XDD is the crux of this implementation.  An XDD is a decision
// diagram that is conceptually similar to a BDD.  The main difference
// is that while BDDs make decisions based on one bit in the input and
// therefore have two outgoing arcs, XDDs make decisions based on a
// block of B bits, and therefore they have 2^B outgoing arcs.  An XDD
// represents a database D.  Each node in the diagram focuses on a
// block of bits in the query Q.  For each value of that block of
// bits, the node points to either (1) another node that focuses on
// the following block of bits, (2) the TRUE node, or (3) the FALSE
// node.
//
// First, we define the terminal (constant) nodes True and False
//
static xdd_node * const XDDFalse = 0;
static xdd_node TrueNode(0);
static xdd_node * const XDDTrue = &TrueNode;

// an xdd_node_context is an auxilary data structure used to annotate
// nodes while building the XDD.  A node context is defined by (1) the
// position of the block of bits (i.e., variables in BDD terminology)
// tested by this node, and (2) the set of candidate sets that remain
// "matchable" by the query set up to this node.
//
struct xdd_node_context {
    const unsigned int block_pos;
    const FABitvector matchable; 
    xdd_node * node;

    xdd_node_context(batch_allocator & mem, unsigned int p, const bitvector & m) 
	: block_pos(p), matchable(mem, m), node(0) {}
};

// xdd_node_map is the data structure we use to store and retrieve XDD
// nodes while building the xdd.  xdd_node_map stores node "contexts"
// (xdd_node_context) and it consists of two interdependent data
// structures: (1) an array of indexes of matchable sets (root[]),
// where each index is implemented as a patricia trie; and (2) a queue
// of contexts (actually an array of queues).  Each newly created
// context is appended to the queue, while the XDD-building algorithm
// processes elements from the head of the queue.
//
// Therefore, the two primary methods we support are:
//
//    xdd_node_context * add();
//    xdd_node_context * pop_first();
//
// See bx_filter::xdd_build() for more details.
//
// For each block position bp, we maintain an index of matchable sets.
// The index is implemented as a patricia trie over the sets
// themselves, which are represented as bitvectors.  See add() for
// details.
// 
class xdd_node_map {
public:
    xdd_node_map();
    ~xdd_node_map();

    xdd_node_context * add(unsigned int p, const bitvector & m);
    xdd_node_context * pop_first();

#ifdef DEBUG_OUTPUT
    unsigned int nodes_count;
#endif
private:
    class pq_node: public xdd_node_context {
    public:
	pq_node * next;
	int pos;
	pq_node * left;
	pq_node * right;

	pq_node(batch_allocator & mem, 
		unsigned int bp, 
		const bitvector & k, 
		int p)
	    : xdd_node_context(mem, bp, k), next(0), pos(p), 
	      left(this), right(this) { }
    };
    // WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION defines an xdd_node_map
    // that progressively deallocates node context data (indexes,
    // queues, etc.) as it progresses through the levels.  This might
    // be useful in tight, border-line memory situations.  The default
    // is to deallocate all the temporary context information at once,
    // when the nodemap is destroyed, which happens at the end of
    // xdd_build()
    //
#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
    batch_allocator mem[bloom_filter<>::B8Size];
#else
    batch_allocator mem;
#endif
    pq_node * root[bloom_filter<>::B8Size];
    pq_node * queue[bloom_filter<>::B8Size];
    pq_node * enqueue(pq_node * n);
    unsigned int current_block_pos;
};

xdd_node_map::xdd_node_map() : 
#ifdef DEBUG_OUTPUT
    nodes_count(0), 
#endif
    current_block_pos(0) {
    for(unsigned int i = 0; i < bloom_filter<>::B8Size; ++i) {
	root[i] = queue[i] = 0;
    }
}

xdd_node_map::~xdd_node_map() { 
#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
    while(current_block_pos < bloom_filter<>::B8Size) {
	mem[current_block_pos].clear();
	++current_block_pos;
    }
#else
    mem.clear();
#endif
}

xdd_node_map::pq_node * xdd_node_map::enqueue(pq_node * n) {
#ifdef DEBUG_OUTPUT
    ++nodes_count;
    if (nodes_count % 1000 == 1) {
	std::cout << "Nodes: " << (nodes_count / 1000) 
		  << "K    \r" << std::flush;
    }
#endif
    n->next = queue[n->block_pos];
    queue[n->block_pos] = n;
    return n;
}

xdd_node_context * xdd_node_map::pop_first() {
    while(current_block_pos < bloom_filter<>::B8Size) {
	if (queue[current_block_pos]) {
	    pq_node * res = queue[current_block_pos];
	    queue[current_block_pos] = res->next;
	    return res;
	} 
#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
	mem[current_block_pos].clear();
#endif
	++current_block_pos;
    }
    return 0;
}

// we look up the node map to find a node that focuses on the block at
// position pb, and that represents the given set M of matchable
// database sets.  This function returns a an existing node context if
// the matchable set is already in the index, or a new context
// otherwise.  In the latter case, the newly-created context is also
// appended to the queue at position pb.
//
// This patricia search/insert algorithm is based on the same
// representation and algorithm of patricia.h (see comment in there).
// However, this one is a bit different in that we do not represent
// the root of the patricia trie with a "NULLItem" node.  Instead, we
// use a root pointer (root[pb]), initially NULL, and pointing
// directly to a valid node.  The primary motivation for this is
// precisely so that we don't have to deal with a NULLItem.  A minor
// side-effect is that we waste one less node and we have exactly one
// node per element in the index.  The down side is that we must
// complicate the algorithm just a bit to deal with all "border"
// cases.
// 
xdd_node_context * xdd_node_map::add(unsigned int bp, const bitvector & M) {
    if (root[bp] == 0) {	// 1) empty index: insert immediately
#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
	return enqueue(root[bp] = new (mem[bp]) pq_node(mem[bp], bp, M, -1));
#else
	return enqueue(root[bp] = new (mem) pq_node(mem, bp, M, -1));
#endif
    }
    int pos;
    register pq_node * n = root[bp];

    if (n->pos >= 0) {		// here we search the patricia trie:
	do {			// we follow the trie as long as there
	    pos = n->pos;	// is a "lower" node
	    n = M[pos] ? n->right : n->left;
	} while (n->pos > pos);
    }
    if (n->matchable == M) {	// if we found the key (M) we simply
	return n;		// return it, otherwise we look for
    }				// the first position in which the
				// current node differs from the key
    for(pos = 0; n->matchable[pos] == M[pos]; ++pos);

    n = root[bp];		// here we insert the new node:
    if (n->pos < 0) {
	n->pos = pos;		// 2) special case: only one node in index

#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
	pq_node * new_node = new (mem[bp]) pq_node(mem[bp], bp, M, pos);
#else
	pq_node * new_node = new (mem) pq_node(mem, bp, M, pos);
#endif
	if (M[pos]) {
	    n->right = new_node;
	} else {
	    new_node->right = n;
	    root[bp] = new_node;
	}
	return enqueue(new_node);
    }
    // 3) two or more nodes already in the index.
    register pq_node ** source_p = &(root[bp]);
    int source_pos = -1;
    while (n->pos < pos && source_pos < n->pos) {
	source_pos = n->pos;
	source_p = M[source_pos] ? &(n->right) : &(n->left);
	n = *source_p;
    }
#ifdef WITH_XDDNODEMAP_INCREMENTAL_DEALLOCATION
    *source_p = new (mem[bp]) pq_node(mem[bp], bp, M, pos);
#else
    *source_p = new (mem) pq_node(mem, bp, M, pos);
#endif
    if (M[pos]) {
	(*source_p)->left = n;
    } else {
	(*source_p)->right = n;
    }
    return enqueue(*source_p);
}

// shortcut_forward() attempts to skip ininfluential positions,
// thereby avoiding the creation and processing of useless XDD nodes.
// An ininfluential (block) position is one where all the matchable
// filters have empty blocks
// 
static unsigned int shortcut_forward(unsigned int pos, 
				     const bx_filter * l,
				     const bitvector & M) {
    while(pos < bloom_filter<>::B8Size) {
	for(const bx_filter * curs = l; (curs); curs = curs->next_tmp) {
	    if (M[curs->fpos] && curs->b.bv8(pos) != 0) {
		return pos;
	    }
	}
	++pos;
    }
    return pos;
}
static unsigned int shortcut_forward(unsigned int pos, const bx_filter * l) {
    while(pos < bloom_filter<>::B8Size) {
	for(const bx_filter * curs = l; (curs); curs = curs->next_tmp) {
	    if (curs->b.bv8(pos) != 0) {
		return pos;
	    }
	}
	++pos;
    }
    return pos;
}

// only_trailing_zeros() tells us whether bitset b has only trailing
// zero-bits starting from bit position pos to the end.  This is used
// to try to shortcut to the TRUE node.
// 
static inline bool only_trailing_zeros(const bloom_filter<> & b, unsigned int pos) {
    while (pos < bloom_filter<>::B8Size) {
	if (b.bv8(pos)) return false;
	++pos;
    }
    return true;
}

// xdd_node_map::add() returns an xdd_node_context.  This context contains
// a pointer to an xdd_node.  This is either a valid pointer, if the
// context was already in the index, or NULL if this is a new node.
// Therefore, this helper function creates the new xdd_node when
// needed.
//
static inline xdd_node * add_node(xdd_node_map & q, 
				  unsigned int pos, const bitvector & M, 
				  batch_allocator & mem) {
    xdd_node_context * c = q.add(pos, M);
    if (c->node) {
	return c->node;
    } else {
	return c->node = new (mem) xdd_node(pos);
    }
}

// this is the main procedure that builds the XDD.  We start off from
// the root node of the XDD, which looks at block 0 and includes all
// the sets (filters) in its "matchable" set.  For each node, at block
// poisition pos, we scan each value v of the block at position pos,
// and we find which sets would remain "matchable" if that value v
// were present at the given position in the query.  If no sets remain
// matchable, we connect the v-th outgoing link to the FALSE node.
// Otherwise, if some sets remain matchable and that was the last
// block, then we connect the v-th link to the true node.  Otherwise
// we connect the v-th link to another node that examines the
// following block, and that has the resulting set of matchables.
//
void bx_predicate::build_xdd(batch_allocator & mem) {
    // first of all we must figure out the size of the filter list.
    // Also, we need to assign filter positions.  We do this by simply
    // scanning the list.  We could do this 100 times better, but that
    // means complicating the filter list data structure in a way that
    // is completely unnecessary.
    //
    unsigned int flist_size = 0;
    bx_filter * curs = flist;
    for(curs = flist; curs; curs = curs->next)
	++flist_size;

    DBG("buiding xdd for predicate " << id << " with " 
	<< flist_size << " filters" << std::endl);
    xdd_node_map q;

    // we push the root node on the queue
    bitvector M(flist_size, true);
    xdd = add_node(q, 0, M, mem);

    xdd_node_context * qi; // current node (context), or "queue iterator"

#ifdef DEBUG_OUTPUT
    unsigned int prev_pos = 0xffffffff;
#endif
    while((qi = q.pop_first())) { // for each node in the queue
	const bitvector & qi_matchable = qi->matchable;
	unsigned int qi_pos = qi->block_pos;
	xdd_node & qi_node = *qi->node;

#ifdef DEBUG_OUTPUT
	if (prev_pos != qi_pos) {
	    prev_pos = qi_pos;
	    std::cout << "pos=" << qi_pos << " nodes=" 
		      << q.nodes_count << std::endl;
	}
#endif
	// here we test all the values v=0..255 against all the
	// qi_pos-th block of each filter that is still matchable.  We
	// know which filters are still matchable from qi_matchable.
	//
	// We try to shortcut the evaluation of each individual filter
	// block by computing the union of all the blocks and by
	// checking whether v (1) covers, (2) simply intersect, or (3)
	// doesn't intersect the union of all the blocks.  
	//
	// In (1) we skip the iteration and link to the next node that
	// preserves the "matchable" set.  In (2) we go through the
	// iteration to see which filters stay matchable. And in (3)
	// we skip the loop and link the node to the FALSE node.
	//
	// We actually unroll the loop as follows: first we check v ==
	// 0, then v == 255, and then v == 1..254.  We compute the
	// "block union" during our first iteration (v == 0).
	//
	unsigned char block_union = 0;
	unsigned char v = 0;
	bx_filter * flist_tmp = 0;
	bx_filter ** cursp;
	cursp = &flist_tmp;
	unsigned int next_pos;

	// v == 0
	M = qi_matchable;
	curs = flist;
	while(curs != 0 && M.get_count() > 0) { 
	    if (M[curs->fpos]) {
		*cursp = curs;
		curs->next_tmp = 0;
		cursp = &(curs->next_tmp);
		block_union |= curs->b.bv8(qi_pos);

		if (curs->b.bv8(qi_pos) != 0) {
		    M.clear(curs->fpos);
		}
	    }
	    curs = curs->next;
	}
	if (M.get_count() == 0) {
	    // no more matchable filters with v (here 0), therefore we
	    // simply point to the False node.
	    qi_node.table[0] = XDDFalse;
	} else {
	    // some filters are still matchable
	    next_pos = shortcut_forward(qi_pos + 1, flist_tmp, M);
	    if (next_pos < bloom_filter<>::B8Size) {
		// next_pos isn't the last position (BVSize),
		// then we connect the v-th slot to the xdd_node
		// at pos next_pos that has a matchable set =
		// matchable
		qi_node.table[0] = add_node(q, next_pos, M, mem);
	    } else {
		// otherwise we reached our target, and therefore
		// we link the v-th slot with the True node
		qi_node.table[0] = XDDTrue;
	    }
	}
	// v == 255
	for(curs = flist_tmp; curs != 0; curs = curs->next_tmp) {
	    if (only_trailing_zeros(curs->b, qi_pos + 1)) {
		qi_node.table[255] = XDDTrue;
		goto value_loop;
	    }
	}
	if (M.get_count() == qi_matchable.get_count()) {
	    qi_node.table[255] = qi_node.table[0];
	} else {
	    next_pos = shortcut_forward(qi_pos + 1, flist_tmp);
	    if (next_pos < bloom_filter<>::B8Size) {
		qi_node.table[255] = add_node(q, next_pos, qi_matchable, mem);
	    } else {
		qi_node.table[255] = XDDTrue;
	    }
	}

    value_loop:			// v == 1..254
	for(v = 1; v < 255; ++v) {
	    if (covers(v, block_union)) {
		qi_node.table[v] = qi_node.table[255];
	    } else if (!intersect(v, block_union)) {
		qi_node.table[v] = qi_node.table[0];
	    } else {
		M = qi_matchable;
		for(curs = flist_tmp; curs != 0; curs = curs->next_tmp) {
		    if (!covers(v, curs->b.bv8(qi_pos))) {
			// value v does not cover (in the sense of
			// the Bloom filter) the i-th block of the
			// fpos-th filter, so we remove the
			// fpos-th filter from the "matchable" set
			//
			M.clear(curs->fpos);
		    } else if (only_trailing_zeros(curs->b, qi_pos + 1)) {
			goto link_to_true;
		    }
		}
		if (M.get_count() == 0) { 
		    // no more matchable filters
		    qi_node.table[v] = XDDFalse;
		} else { 
		    // some filters are still matchable
		    next_pos = shortcut_forward(qi_pos + 1, flist_tmp, M);
		    if (next_pos < bloom_filter<>::B8Size) {
			qi_node.table[v] = add_node(q, next_pos, M, mem);
		    } else {
		    link_to_true:
			qi_node.table[v] = XDDTrue;
		    }
		}
	    }
	}
    }
    DBG("done buiding xdd.  Total xdd nodes = " << q.nodes_count << std::endl);
}

// the matching function for a predicate represented as an XDD is
// trivial: starting at the root of the XDD, we simply look at the
// current block in the input set 
//
bool bx_predicate::matches(const bloom_filter<> & b) const {
    xdd_node * n = xdd; 
    if (!n) return false;

    DBG("Matches: ");
    for (;;) {
	DBG(n->pos << "-->");
        n = n->table[b.bv8(n->pos)];
        if (n == XDDFalse) {
	    DBG("False" << std::endl);
	    return false;
	} else if (n == XDDTrue) {
	    DBG("True" << std::endl);
	    return true;
	} 
    }
}

void bx_table::consolidate() { }

void bx_table::match(const siena::Message & m, siena::MatchHandler & h) const {
    const bx_predicate * p = plist;
    if (!p) return;

    siena::Message::Iterator * i = m.first();
    if (!i) return;

    bloom_filter<> b;

    TIMER_PUSH(bloom_encoding_timer);

    do {
	if (i->type() != siena::ANYTYPE)
	    encode_attribute(b, i); 
    } while (i->next());
    delete(i);
    DBG("encoded message: " << std::endl << b << std::endl);

    TIMER_POP();

    TIMER_PUSH(match_timer);

    do {
	if (p->matches(b)) {

	    TIMER_PUSH(forward_timer);

	    bool output_result = h.output(p->id);

	    TIMER_POP();

	    if (output_result)
		break;
	}
    } while ((p = p->next));
    TIMER_POP();
}

void bx_table::clear() {
    memory.clear();
    plist = 0;
}
void bx_table::clear_recycle() {
    memory.recycle();
    plist = 0;
}

size_t bx_table::allocated_bytesize() const {
    return memory.allocated_size();
}

size_t bx_table::bytesize() const {
    return memory.size();
}

} // end namespace siena_impl

siena::BXTable * siena::BXTable::create() {
    return new siena_impl::bx_table();
}


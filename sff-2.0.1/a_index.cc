// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
//                2011 Antonio Carzaniga
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

#ifndef WITH_A_INDEX_USING_TST
#include <climits>
#include <cstring>
#endif

#include <cassert>

#include "allocator.h"
#include "a_index.h"

namespace siena_impl {

class fwd_attribute;


#ifdef WITH_A_INDEX_USING_TST
// 
// This is an implementation of an attribute table based on a ternary
// search trie (TST).  I am not too smart about the balancing of the
// TST, but I do balance each per-character BST that composes the TST.
//
class a_index_node {
public:
    a_index_node(int xc) 
	: data(0), c(xc), left(0), middle(0), right(0) {};

    fwd_attribute * data;
    int c;
    a_index_node * left;
    a_index_node * middle;
    a_index_node * right;
};

void a_index::clear() {
    //
    // I don't actually deallocate stuff here since I'm assuming that
    // ftmemory will be deallocated everything at once
    //
    for(int i = 0; i < 256; ++i)
	roots[i] = 0;
}

static const int END_CHAR = -1;

//
// these two algorithms are taken more or less directly from
// R. Sedgewick's "Algorithms in C" 3rd Ed. pp 638--639.
//
// Two differences:
//
// 1) I use an aray of 256 root pointers indexed by the first
// character of the string, instead of a single root.  I guess this
// modification doesn't allow me to store (and retrieve) the empty
// string, which is fine for my application.
//
// 2) These version are not recursive.
//
fwd_attribute ** a_index::insert(const char * s, const char * end, 
				 batch_allocator & ftmemory) {
    assert(s != end);
    a_index_node * pp = 0;
    a_index_node ** p = &(roots[static_cast<unsigned char>(*s++)]);
    while (*p != 0) {
	pp = *p;
	if (s == end) {
	    if (pp->c == END_CHAR) return &(pp->data);
            p = &(pp->left);
	} else if (*s == pp->c) {
	    ++s;
            p = &(pp->middle);
        } else if (*s < pp->c) {
            p = &(pp->left);
	} else {
            p = &(pp->right);
	}
    }

    for (;;) {
	if (s == end) {
	    *p = new (ftmemory)a_index_node(END_CHAR);
	    return &((*p)->data);
	}
	*p = new (ftmemory)a_index_node(*s);
	pp = *p;
	++s;
        p = &(pp->middle);
    }
}

const fwd_attribute * a_index::find(const char * s, 
				    const char * end) const {
    assert(s != end);
    register const a_index_node * p = roots[static_cast<unsigned char>(*s++)];
    while (p != 0) {
	if (s == end) {
	    if (p->c == END_CHAR) return p->data;
            p = p->left;
	} else if (*s < p->c) {
            p = p->left;
	} else if (*s == p->c)  {
	    ++s;
            p = p->middle;
        } else {
            p = p->right;
	}
    }
    return 0;
}

static void right_rotate(a_index_node ** pp) {
    assert(*pp != 0 && (*pp)->left != 0);

    a_index_node * p_left = (*pp)->left;
    a_index_node * p_left_right = p_left->right;
    (*pp)->left = p_left_right;
    p_left->right = *pp;
    *pp = p_left;
}

static unsigned int tree_to_vine(a_index_node ** rootp) {
    unsigned int size = 0;
    while(*rootp != 0) {
	while((*rootp)->left != 0) 
	    right_rotate(rootp);
	++size;
	rootp = &((*rootp)->right);
    }
    return size;
}

static void left_rotate(a_index_node ** pp) {
    assert(*pp != 0 && (*pp)->right != 0);

    a_index_node * p_right = (*pp)->right;
    a_index_node * p_right_left = p_right->left;
    (*pp)->right = p_right_left;
    p_right->left = *pp;
    *pp = p_right;
}

static void compress(a_index_node ** pp, unsigned int count) {
    while (count > 0) {
	left_rotate(pp);
	pp = &((*pp)->right);
	--count;
    }
}

static unsigned int sup_power_of_two(unsigned int x) {
    // return the largest power of 2 that is less than or equal to x
    unsigned int y;
    for(;;) {
	y = x & (x - 1);	// clear the least significant bit set
	if (y == 0)
	    return x;
	x = y;
    }
}

static void vine_to_balanced_tree(a_index_node ** rootp, unsigned int size) {
    unsigned int leaf_count = size + 1 - sup_power_of_two(size + 1);
    compress(rootp, leaf_count);
    size -= leaf_count;
    while(size > 1) 
	compress(rootp, size /= 2);
}

//
// This is the BST balancing algorithm described in "Tree Rebalancing
// in Optimal Time and Space" by Q. F. Stout and B. L. Warren,
// Communications of the ACM (CACM) Volume 29 Issue 9, Sept. 1986
//
static void balance_bst(a_index_node ** rootp) {
    unsigned int size;
    size = tree_to_vine(rootp);
    vine_to_balanced_tree(rootp, size);
}

static void balance_tst(a_index_node ** rootp);

static void apply_balance_tst(a_index_node * root) {
    if (root->middle)
	balance_tst(&(root->middle));
    if (root->left)
	apply_balance_tst(root->left);
    if (root->right)
	apply_balance_tst(root->right);
}

static void balance_tst(a_index_node ** rootp) {
    // assumes *rootp != 0
    balance_bst(rootp);
    apply_balance_tst(*rootp);
}

void a_index::consolidate() {
    for(int i = 0; i < 256; ++i) 
	if (roots[i])
	    balance_tst(&(roots[i]));
}

#else // we use a_PATRICIA trie
//
// This is an attribute table based on a PATRICIA trie, but is
// extended so as to be able to support binary strings, that is,
// strings that are not delimited by a terminator character that
// therefore can not appear within a string.
// 
// The main idea here is that each bit can have *three* values: ZERO,
// ONE, and OUT-OF-BOUNDS.  Therefore, each node in the trie must have
// three pointers, one for each of the three bit values.
//
// Beyond that, this is pretty much the same PATRICIA trie as
// described in R.Sedgewick "Algorithms in C" 3rd Ed., pp. 623--627.
// 
enum bit_value_t {
    BIT_ZERO		= 0,
    BIT_ONE		= 1,
    BIT_OUT_OF_BOUNDS	= 2
};

struct a_index_node {
    a_index_node(const char * b, const char * e, int p) throw()
	: key_begin(b), key_end(e), pos(p), data(0) {
	next[BIT_ZERO] = next[BIT_ONE] = next[BIT_OUT_OF_BOUNDS] = this;
    };
    const char *	key_begin;
    const char *	key_end;
    const int		pos;
    a_index_node * 	next[3];
    fwd_attribute *	data;
};

// INVALID_POS:
// 
// node->pos == INVALID_POS indicates that no bit must be tested for
// this node.  Recall that in this (i.e., Sedgewick's) PATRICIA trie,
// there are no terminal nodes, and therefore each node carries a key.
// More specifically, each terminal node is also used as an "internal"
// node by indicating the position at which its key differs from that
// of another node.  Therefore, when the first node (i.e., the first
// key) is inserted in the trie, there is no other key to compare
// against, and therefore the bit position at which two keys differ is
// undefined.  No other node/key carries an INVALID_POS position.
//
static const int INVALID_POS = INT_MAX;

static int find_first_diff_bit(const char * a, const char * a_end, 
			       const char * b, const char * b_end) {
    // Finds the first bit position at which two strings differ.  It
    // is supposed to be called with two different strings.
    int res = 0;
    while(a != a_end && b != b_end && *a == *b) {
	res += CHAR_BIT;
	++a;
	++b;
    }

    // assert(a != a_end || b != b_end)
    if (a == a_end || b == b_end)
	return res;

    // assert(*a != *b)
    char axb = *a ^ *b;
    while((axb & 1) == 0) {
	++res;
	axb >>= 1;
    }
    return res;
}

static inline bool string_equal(const char * a_begin, const char * a_end, 
				const char * b_begin, const char * b_end) {
    if (a_end - a_begin != b_end - b_begin) 
	return false;
    return (memcmp(a_begin, b_begin, a_end - a_begin) == 0);
}

static inline bit_value_t bit_at_position(const char * begin, const char * end, 
					  int pos) {
    int char_pos = pos / CHAR_BIT;
    if (char_pos < end - begin) {
	return (begin[char_pos] & (1 << (pos % CHAR_BIT))) ? BIT_ONE : BIT_ZERO;
    } else {
	return BIT_OUT_OF_BOUNDS; 
    }
}

static a_index_node * new_node(const char * begin, const char * end, int pos,
			       batch_allocator & mem) {
    size_t slen = end - begin;
    char * s = new (mem) char[slen];
    memcpy(s, begin, slen);
    return new (mem) a_index_node(s, s + slen, pos);
}

fwd_attribute ** a_index::insert(const char * begin, const char * end, 
				 batch_allocator & mem) {
    a_index_node * n = root;
    if (!n) {
	root = new_node(begin, end, INVALID_POS, mem);
	return &(root->data);
    } 

    int prev_pos;
    do {
	prev_pos = n->pos;
	if (n->pos != INVALID_POS) {
	    n = n->next[bit_at_position(begin, end, n->pos)];
	} else 
	    break;
    } while (n->pos > prev_pos);

    if (string_equal(begin, end, n->key_begin, n->key_end))
	return &(n->data);

    int diff_pos = find_first_diff_bit(begin, end, n->key_begin, n->key_end);

    a_index_node ** target = &root;
    prev_pos = -1;
    while ((*target)->pos <= diff_pos) {
	prev_pos = (*target)->pos;
	if ((*target)->pos != INVALID_POS) {
	    target = &((*target)->next[bit_at_position(begin, end, (*target)->pos)]);
	} else 
	    break;
	if ((*target)->pos <= prev_pos)
	    break;
    }
    // assert((*target)->pos < diff_pos 
    n = new_node(begin, end, diff_pos, mem);
    n->next[bit_at_position((*target)->key_begin, (*target)->key_end, diff_pos)] = *target;
    *target = n;

    return &(n->data);
}

const fwd_attribute * a_index::find(const char * begin, 
				    const char * end) const {
    a_index_node * n = root;
    if (!n) return 0;

    int prev_pos;

    do {
	prev_pos = n->pos;
	if (n->pos != INVALID_POS) {
	    n = n->next[bit_at_position(begin, end, n->pos)];
	} else 
	    break;
    } while (n->pos > prev_pos);

    if (string_equal(begin, end, n->key_begin, n->key_end))
	return n->data;
    else
	return 0;
}

void a_index::clear() {
    root = 0;
}
void a_index::consolidate() {}

#endif // WITH_A_INDEX_USING_TST

} // end namespace siena_impl

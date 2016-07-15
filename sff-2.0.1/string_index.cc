// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002-2003 University of Colorado
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

#include <cassert>

#include <siena/forwarding.h>
#include "allocator.h"

#include "fwd_table.h"
#include "pointers_set.h"
#include "string_index.h"

#include "timers.h"

namespace siena_impl {

static const int EndChar = -1;

typedef unsigned char mask_t;

static const mask_t EQ_Mask = 1;
static const mask_t SF_Mask = 2;
static const mask_t LT_Mask = 4;
static const mask_t GT_Mask = 8;
static const mask_t NE_Mask = 16;
static const mask_t PF_Mask = 32;
static const mask_t SS_Mask = 64;

static const mask_t Complete_Mask = EQ_Mask | SF_Mask | LT_Mask | GT_Mask | NE_Mask;

//
// this algorithm is taken more or less directly from R. Sedgewick's
// "Algorithms in C" 3rd Ed. pp 638--639.  The only difference is the
// addition of my subtree mask.
//
string_index::node * string_index::insert_complete(const char * s, const char * end, 
						   batch_allocator & ftmemory,
						   const mask_t mask) {
    string_index::node * pp = 0;
    string_index::node ** p = &root;
    while (*p != 0) {
	pp = *p;
	pp->mask |= mask;
	if (s == end) {
	    if (pp->c == EndChar) return pp;
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
	if (s == end) 
	    return *p = new (ftmemory)string_index::node(EndChar, pp, mask);
	*p = new (ftmemory)string_index::node(*s, pp, mask);
	pp = *p;
	++s;
        p = &(pp->middle);
    }
}

string_index::node * string_index::insert_partial(const char * s, const char * end,
						  batch_allocator & ftmemory,
						  const mask_t mask) {
    // precondition s != end
    string_index::node * pp = 0;
    string_index::node ** p = &root;
    while (*p != 0) {
	pp = *p;
	pp->mask |= mask;
	if (*s == pp->c) {
	    if (++s == end) return pp;
            p = &(pp->middle);
        } else if (*s < pp->c) {
            p = &(pp->left);
	} else {
            p = &(pp->right);
	}
    }
    for (;;) {
	*p = new (ftmemory)string_index::node(*s, pp, mask);
	pp = *p;
        if (++s == end) return pp;
        p = &(pp->middle);
    }
}

string_index::node * string_index::backtrack_next(register string_index::node * p, 
						  register const mask_t mask) {
    register string_index::node * pp;
    while ((pp = p->up) != 0) {
	if (p == pp->left) goto back_from_left;
	if (p == pp->middle) goto back_from_middle;
	if (p == pp->right) goto back_from_right;

	assert(false);		// this should NEVER happen!

    back_from_left:
	if ((pp->middle) && (pp->middle->mask & mask)) 
	    return pp->middle;
    back_from_middle:
	if ((pp->right) && (pp->right->mask & mask)) 
	    return pp->right;
    back_from_right:
	p = pp;
    }
    return 0;
}
//
// returns the node corresponding to the next complete string that is
// lexicographically greater than the (complete) string represented by
// the given node p
//
string_index::node * string_index::next(register string_index::node * p, 
					register const mask_t mask) {
    if ((p->left) && (p->left->mask & mask)) {
	p = p->left;
    } else if ((p->middle) && (p->middle->mask & mask)) {
	p = p->middle;
    } else if ((p->right) && (p->right->mask & mask)) {
	p = p->right;
    } else {
	if ((p = backtrack_next(p, mask)) == 0)
	    return 0;
    }
    for (;;) {
	if ((p->left) && (p->left->mask & mask)) {
	    p = p->left;
        } else if (p->c == EndChar) {
	    return p; 
	} else if ((p->middle) && (p->middle->mask & mask)) {
            p = p->middle;
	} else if ((p->right) && (p->right->mask & mask)) {
	    p = p->right;
	} else {
	    assert(false);	// this should NEVER happen!
	}
    }
}

string_index::node * string_index::prev(register string_index::node * p,
					register const mask_t mask) {
    // first we backtrack
    register string_index::node * pp = p;
    while ((p = p->up)) {
	if (pp == p->left) goto back_from_left;
	if (pp == p->middle) goto back_from_middle;
	if (pp == p->right) goto back_from_right;
	assert(false);

    back_from_right:
	if (p->c == EndChar) return p;
	if ((p->middle) && (p->middle->mask & mask)) {
	    p = p->middle;
	    break;
	}
    back_from_middle:
	if ((p->left) && (p->left->mask & mask)) {
	    p = p->left;
	    break;
	}
    back_from_left:
	pp = p;
    }
    if (p == 0) 
	return 0;
    // then we move forward keeping to the rightmost path
    for(;;) {
	if ((p->right) && (p->right->mask & mask)) {
	    p = p->right;
	} else if ((p->middle) && (p->middle->mask & mask)) {
	    p = p->middle;
	} else if (p->c == EndChar) {
	    return p; 
	} else if ((p->left) && (p->left->mask & mask)) {
	    p = p->left;
	} else {
	    assert(false);	// this should NEVER happen!
	}
    }
}

const string_index::node * string_index::last(register const string_index::node * pp,
					      register const mask_t mask) {
    if (pp == 0 || (pp->mask & mask) == 0) return 0;
    for (;;) {
	if ((pp->right) && (pp->right->mask & mask)) {
	    pp = pp->right;
	} else if ((pp->middle) && (pp->middle->mask & mask)) {
            pp = pp->middle;
	} else if (pp->c == EndChar) {
	    return pp;
	} else if ((pp->left) && (pp->left->mask & mask)) {
	    pp = pp->left;
	} else {
	    assert(false);	// this should NEVER happen!
	}
    }
}

const string_index::node * string_index::upper_bound(register const string_index::node * p,
						     register const char * begin,
						     register const char * end,
						     register const mask_t mask) {
    //
    // precondition root != 0
    //
    if ((p->mask & mask) == 0) return 0;
    register const string_index::node * ub;
    if (begin == end) {
	ub = p;
    } else {
	register char s = *begin;
	ub = 0;
	for (;;) {
	    if (s == p->c) {
		if ((p->right) && (p->right->mask & mask))
		    ub = p->right;

		if ((p->middle) && (p->middle->mask & mask)) {
		    p = p->middle;
		    if (++begin == end) {
			if (p->c == EndChar) {
			    if ((p->right) && (p->right->mask & mask)) {
				ub = p->right;
			    } 
			} else {
			    ub = p;
			}
			break;
		    } 
		    s = *begin;
		} else {
		    break;
		}
	    } else if (s < p->c) {
		if ((p->middle) && (p->middle->mask & mask)) {
		    ub = p->middle;
		} else if ((p->right) && (p->right->mask & mask)) {
		    ub = p->right;
		}
		if ((p->left) && (p->left->mask & mask)) {
		    p = p->left;
		} else {
		    break;
		}
	    } else { 		// (s > pp->c)
		if ((p->right) && (p->right->mask & mask)) {
		    p = p->right;
		} else {
		    break;
		}
	    }
	}
	if (ub == 0) return 0;
    }
    while (ub->c != EndChar) {
	if ((ub->left) && (ub->left->mask & mask)) {
	    ub = ub->left;
	} else if ((ub->middle) && (ub->middle->mask & mask)) {
	    ub = ub->middle;
	} else if ((ub->right) && (ub->right->mask & mask)) {
	    ub = ub->right;
	} else {
	    assert(false);	// this should NEVER happen!
	}
    }
    return ub;
}

void string_index::insert_between(complete_descr * d, 
				  string_index::node * p, string_index::node * n) {
    if (p == 0) {
	d->next_gt = 0;
    } else {
	complete_descr * pd = p->c_data;
	d->next_gt = (pd->gt || pd->ne) ? pd : pd->next_gt;
	if (d->lt != 0 || d->ne != 0) {
	    for (;;) {
		pd->next_lt = d;
		if (pd->lt != 0 || pd->ne != 0) break;
		p = prev(p, Complete_Mask);
		if (p == 0) break;
		pd = p->c_data;
	    }
	}
    }

    if (n == 0) {
	d->next_lt = 0;
    } else {
	complete_descr * nd = n->c_data;
	d->next_lt = (nd->lt || nd->ne) ? nd : nd->next_lt;
	if (d->gt != 0 || d->ne != 0) {
	    for (;;) {
		nd->next_gt = d;
		if (nd->gt != 0 || nd->ne != 0) break;
		n = next(n, Complete_Mask);
		if (n == 0) break;
		nd = n->c_data;
	    }
	}
    }
}

fwd_constraint * string_index::add_eq(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_complete(v.begin, v.end, ftmemory, EQ_Mask);
    complete_descr * d;
    if (np->c_data == 0) {
	d = new (ftmemory)complete_descr();
	np->c_data = d;
	d->eq = new (ftmemory)fwd_constraint();
	d->ne = 0;
	d->lt = 0;
	d->gt = 0;
	d->sf = 0;
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
	return d->eq;
    }
    d = np->c_data;
    if (d->eq == 0) {
	d->eq = new (ftmemory)fwd_constraint();
    }
    return d->eq;
}

fwd_constraint * string_index::add_sf(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_complete(v.begin, v.end, ftmemory, SF_Mask);
    complete_descr * d;
    if (np->c_data == 0) {
	d = new (ftmemory)complete_descr();
	np->c_data = d;
	d->eq = 0;
	d->ne = 0;
	d->lt = 0;
	d->gt = 0;
	d->sf = new (ftmemory)fwd_constraint();
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
	return d->sf;
    }
    d = np->c_data;
    if (d->sf == 0) {
	d->sf = new (ftmemory)fwd_constraint();
    }
    return d->sf;
}

fwd_constraint * string_index::add_ne(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_complete(v.begin, v.end, ftmemory, NE_Mask);
    complete_descr * d;
    if (np->c_data == 0) {
	d = new (ftmemory)complete_descr();
	np->c_data = d;
	d->eq = 0;
	d->ne = new (ftmemory)fwd_constraint();
	d->lt = 0;
	d->gt = 0;
	d->sf = 0;
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
	return 	d->ne;
    }
    d = np->c_data;
    if (d->ne == 0) {
	d->ne = new (ftmemory)fwd_constraint();
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
    }
    return d->ne;
}

fwd_constraint * string_index::add_lt(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_complete(v.begin, v.end, ftmemory, LT_Mask);
    complete_descr * d;
    if (np->c_data == 0) {
	d = new (ftmemory)complete_descr();
	np->c_data = d;
	d->eq = 0;
	d->ne = 0;
	d->lt = new (ftmemory)fwd_constraint();
	d->gt = 0;
	d->sf = 0;
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
	return 	d->lt;
    }
    d = np->c_data;
    if (d->lt == 0) {
	d->lt = new (ftmemory)fwd_constraint();
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
    }
    return d->lt;
}

fwd_constraint * string_index::add_gt(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_complete(v.begin, v.end, ftmemory, GT_Mask);
    complete_descr * d;
    if (np->c_data == 0) {
	d = new (ftmemory)complete_descr();
	np->c_data = d;
	d->eq = 0;
	d->ne = 0;
	d->lt = 0;
	d->gt = new (ftmemory)fwd_constraint();
	d->sf = 0;
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
	return 	d->gt;
    }
    d = np->c_data;
    if (d->gt == 0) {
	d->gt = new (ftmemory)fwd_constraint();
	insert_between(d, prev(np, Complete_Mask), next(np, Complete_Mask));
    }
    return d->gt;
}

fwd_constraint * string_index::add_pf(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_partial(v.begin, v.end, ftmemory, PF_Mask);
    partial_descr * d;
    if (np->p_data == 0) {
	d = new (ftmemory)partial_descr();
	np->p_data = d;
	d->ss = 0;
	d->pf = new (ftmemory)fwd_constraint();
	return d->pf;
    }
    d = np->p_data;
    if (d->pf == 0) {
	d->pf = new (ftmemory)fwd_constraint();
    }
    return d->pf;
}

fwd_constraint * string_index::add_ss(const siena::String & v, 
				      batch_allocator & ftmemory) { 
    string_index::node * np = insert_partial(v.begin, v.end, ftmemory, SS_Mask);
    partial_descr * d;
    if (np->p_data == 0) {
	d = new (ftmemory)partial_descr();
	np->p_data = d;
	d->pf = 0;
	d->ss = new (ftmemory)fwd_constraint();
	return d->ss;
    }
    d = np->p_data;
    if (d->ss == 0) {
	d->ss = new (ftmemory)fwd_constraint();
    }
    return d->ss;
}

fwd_constraint * string_index::add_re(const siena::String & v, 
				      batch_allocator & ftmemory)  {
    throw new siena::BadConstraint ("regular expressions not supported yet");
}

fwd_constraint * string_index::add_any(batch_allocator & ftmemory) { 
    if (any_value == 0)
	any_value = new (ftmemory)fwd_constraint();
    return any_value;
}

bool string_index::match(const siena::String & v, c_processor & cp) const { 
    TIMER_PUSH(string_match_timer);

    if ((any_value) && 		// first we try the any-value constraint
	cp.process_constraint(any_value)) {

	TIMER_POP();

	return true;
    }

    bool result = tst_match (v, cp);

    TIMER_POP();

    return result;
}


bool string_index::tst_match(const siena::String & v, c_processor & cp) const {
    register const string_index::node * p = root;
    if (p == 0) return false;

    register const char * sp = v.begin;
    register char s = *sp;
    
    //
    // We use this set of pointers to keep track of the substring
    // constraints that we have already matched, and therefore that we
    // should simply skip.
    //
    pointers_set ss_cset;
				// in this first pass, we look for all
    while ((p)) {		// constraints: suffix, prefix,
	if (s < p->c) {		// substring and complete constraints
	    p = p->left;
	} else if (s == p->c) {
	    if (p->c != EndChar) {
		if (p->p_data != 0) { // here we have a partial match
		    partial_descr * d = p->p_data;
		    if ((d->pf) && cp.process_constraint(d->pf))
			return true;
		    if ((d->ss) && ss_cset.insert(d->ss)
			&& cp.process_constraint(d->ss))
			return true;
		}
		p = p->middle;
		++sp;
		s = (sp == v.end) ? EndChar : *sp;
	    } else {
		// here we have a complete match
		// it should always be np->data != 0
		complete_descr * d = p->c_data;
		if (d->eq != 0 && cp.process_constraint(d->eq))
		    return true;
		if (d->sf != 0 && cp.process_constraint(d->sf))
		    return true;
		complete_descr * curs;
		for(curs = d->next_lt; curs != 0; curs = curs->next_lt) {
		    if ((curs->lt) && cp.process_constraint(curs->lt))
			return true;
		    if ((curs->ne) && cp.process_constraint(curs->ne))
			return true;
		}
		for(curs = d->next_gt; curs != 0; curs = curs->next_gt) {
		    if ((curs->gt) && cp.process_constraint(curs->gt)) 
			return true;
		    if ((curs->ne) && cp.process_constraint(curs->ne))
			return true;
		}
		goto match_sf_ss;
	    }
        } else {  		// p->c > s
            p = p->right;
	}
    }
    // np is now a middle node that represents the longest
    // (incomplete) prefix of the target
    if ((p = upper_bound(root, v.begin, v.end, Complete_Mask))) {
	complete_descr * d = p->c_data;
	do {
	    if ((d->lt) && cp.process_constraint(d->lt))
		return true;
	    if ((d->ne) && cp.process_constraint(d->ne))
		return true;
	} while((d = d->next_lt));
	d = p->c_data;
	while((d = d->next_gt)) {
	    if ((d->gt) && cp.process_constraint(d->gt))
		return true;
	    if ((d->ne) && cp.process_constraint(d->ne))
		return true;
	}
    } else if ((p = last(root, GT_Mask | NE_Mask))) {
	complete_descr * d = p->c_data;
	do {
	    if ((d->gt) && cp.process_constraint(d->gt))
		return true;
	    if ((d->ne) && cp.process_constraint(d->ne))
		return true;
	} while((d = d->next_gt));
    }

 match_sf_ss:
    const char * spl = v.begin;
    if (spl == v.end) return false; // this takes care of empty strings
    while(++spl != v.end) {
	sp = spl;
	s = *sp;
	p = root;
	while ((p) && (p->mask & (SS_Mask | SF_Mask))) { 
	    if (s < p->c) {
		p = p->left;
	    } else if (s == p->c) {
		if (p->c != EndChar) {
		    if (p->p_data != 0) { // here we have a partial match
			partial_descr * d = p->p_data;
			if (d->ss != 0 && ss_cset.insert(d->ss)
			    && cp.process_constraint(d->ss))
			    return true;
		    }
		    p = p->middle;
		    ++sp;
		    s = (sp == v.end) ? EndChar : *sp;
		} else {
		    // here we have a complete match
		    // it should always be np->data != 0
		    complete_descr * d = p->c_data;
		    if (d->sf != 0 && cp.process_constraint(d->sf))
			return true;
		    break;
		}
	    } else {  		// p->c > s
		p = p->right;
	    }
	}
    }
    return false;
}

} // end namespace siena_impl

// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2003-2004 University of Colorado
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
#ifndef B_TABLE_H
#define B_TABLE_H

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <siena/btable.h>

#include "allocator.h"
#include "bloom_filter.h"

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

/** @brief a filter in a BTable.
 *
 *  a filter is simply a link in a linked list that contains a
 *  Bloom filter, which represents the conjunction of constraints.
 **/
struct b_filter {
    b_filter * next;
    bloom_filter<> b;

    b_filter(b_filter * n, const bloom_filter<> & x) : next(n), b(x) {};
};

/** @brief a predicate in a BTable.
 *
 *  This is an extremely simple data structure.  The predicate is
 *  represented as a linked list of filters.  Then, each predicate
 *  refers directly to its interface id, and is itself a link in a
 *  linked list of predicates, which is what makes up the BTable.
 **/
struct b_predicate {
    b_predicate * next;
    ifid_t id;
    b_filter * flist;

    b_predicate(b_predicate *n, ifid_t i, b_filter * fl) 
	: next(n), id(i), flist(fl) {}

    void add_filter(const siena::Filter * pi, batch_allocator & mem);
    bool matches(const bloom_filter<> & b) const;
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
 *  This implementation is provided primarily for experimental
 *  purposes. Because of the nature of both the encoding of messages
 *  and filters into sets, and the representation of sets with Bloom
 *  filters, this implementation can not provide an exact match.  In
 *  fact, this implementation produces false positives, which means
 *  that a message might be forwarded to more interfaces than the ones
 *  it actually matches.  However, this implementation does not
 *  produce false negatives.  This means that a message will always go
 *  to all the interfaces whose predicate match the message.
 **/
class b_table : public siena::BTable {
public:
    b_table();

    virtual ~b_table();

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
    b_predicate *	plist;
};

} // end namespace siena_impl

#endif

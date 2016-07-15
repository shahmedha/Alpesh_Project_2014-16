// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
#ifndef SIENA_BTABLE_H_INCLUDED
#define SIENA_BTABLE_H_INCLUDED

#include <siena/forwarding.h>

namespace siena {

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
class BTable : public AttributesFIB {
public:
    /** @brief create and initialize a BTable.
     **/
    static BTable * create();
};

/** @brief implementation of the forwarding table based on Bloom
 *         filters.
 *
 *  This implementation extends the \link siena::BTable BTable
 *  algorithm\endlink simply by sorting the Bloom filters within each
 *  predicate.  Bloom filters are sorted by their Hamming weight in
 *  ascending order.  The rationale for this is that the basic BTable
 *  algorithm shortcuts the evaluation (only) when a match is found.
 *  Therefore, it is advantageous to try "smaller" Bloom filters
 *  first, because those are more likely to yield a match.
 **/
class SortedBTable : public BTable {
public:
    /** @brief create and initialize a sorted BTable.
     **/
    static BTable * create();
};

} // end namespace siena

#endif

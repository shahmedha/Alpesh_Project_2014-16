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
#ifndef SIENA_BTRIETABLE_H
#define SIENA_BTRIETABLE_H

#include <cstddef>

#include <siena/forwarding.h>

namespace siena {

/** @brief Implementation of TagsFIB based on an encoding of tag sets
 *         as Bloom filters represented as a trie.
 *
 *  This implementation is based on an encoding of tag sets as Bloom
 *  filters.  Because of the probabilistic nature of Bloom filters,
 *  this implementation can not provide an exact match.  In
 *  particular, this implementation may produce false positives, which
 *  means that a message might be forwarded to more interfaces than
 *  the ones it actually matches.  However, this implementation does
 *  not produce false negatives.  This means that a message will
 *  always go to all the interfaces whose predicate match the message.
 *
 *  The internal structure of BTrieTable is based on a representation
 *  of a set of Bloom filters (i.e., a TagsetList) as a \em trie.
 *  More details are available in the technical documentation within
 *  the source file.
 **/
class BTrieTable : public AttributesFIB {
public:
    /** @brief create and initialize a BTrieTable.
     **/
    static BTrieTable * create();
};

} // end namespace siena

#endif

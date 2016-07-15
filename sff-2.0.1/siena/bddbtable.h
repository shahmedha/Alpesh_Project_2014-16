// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
//  Copyright (C) 2005,2013 Antonio Carzaniga
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
#ifndef SIENA_BDDBTABLE_H_INCLUDED
#define SIENA_BDDBTABLE_H_INCLUDED

#include <siena/forwarding.h>

namespace siena {

/** @brief implementation of the forwarding table based on Bloom
 *         filters. This implementation consolidates each predicate
 *         into a binary decision diagram (BDD) that is then used for
 *         matching.
 *
 *  The idea behind this algorithm is to represent each encoded
 *  predicate as a BDD.  An encoded predicate is constructed as an
 *  M-by-N bit matrix, where M is the chosen size of the Bloom filter,
 *  and N is the number of Bloom filters, each one representing a
 *  conjunction of constraints.  Therefore, each one of the M bits of
 *  the Bloom filters becomes a variable in the BDD (or ZDD).
 *
 *  @see ZDDBTable
 **/
class BDDBTable : public AttributesFIB {
public:
    /** @brief create and initialize a BDDBTable.
     **/
    static BDDBTable * create();
};

/** @brief implementation of the forwarding table based on Bloom
 *         filters. This implementation consolidates each predicate
 *         into a zero-suppressed decision diagram (ZDD), which is
 *         then used for matching.
 *
 *  @see BDDBTable
 **/
class ZDDBTable : public AttributesFIB {
public:
    /** @brief create and initialize a ZDDBTable.
     **/
    static ZDDBTable * create();
};

}; // end namespace siena

#endif

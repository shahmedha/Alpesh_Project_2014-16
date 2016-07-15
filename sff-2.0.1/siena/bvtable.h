// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
#ifndef SIENA_BVTABLE_H
#define SIENA_BVTABLE_H

#include <cstddef>

#include <siena/forwarding.h>

namespace siena {

/** @brief implementation of the forwarding table based on Bloom
 *         filters and a data structure used for matching called XDD.
 *
 *  Like BTable, this implementation is based on encoded filters and
 *  messages.  However, instead of using a linear scan of the set of
 *  encoded filters, this implementation indexes the bitvectors
 *  representing filters with a decision diagram. 
 *
 *  \b WARNING: This implementation is provided primarily for
 *  experimental purposes.  See BTable for details.
 **/
class BVTable : public AttributesFIB {
public:
    /** @brief create and initialize a BVTable.
     **/
    static BVTable * create();
};

} // end namespace siena

#endif

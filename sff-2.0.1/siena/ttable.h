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
#ifndef SIENA_TTABLE_H_INCLUDED
#define SIENA_TTABLE_H_INCLUDED

#include <siena/forwarding.h>

namespace siena {

/** @brief a very simple implementation of a TagsFIB.
 *
 *  Basically, this is a table in which a each interface is associated
 *  with a predicate consisting of a list of tagsets.  Insertion and
 *  matching are essentially sequential algorithms.  In particular,
 *  matching goes through each tagset of each predicate (list) to look
 *  for a subset of the given (message) tagset.
 **/
class TTable : public TagsFIB {
public:
    /** @brief create and initialize a TTable.
     **/
    static TTable * create();
};

} // end namespace siena

#endif

// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2001-2003 University of Colorado
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
#ifndef SIENA_BCTABLE_H_INCLUDED
#define SIENA_BCTABLE_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <siena/btable.h>

namespace siena {

/** @brief implementation of the forwarding table based on Bloom
 *         filters and counting algorithm.
 *
 *  Like BTable, this implementation is based on encoded filters and
 *  messages.  However, instead of using a linear scan of the set of
 *  encoded filters, this implementation uses a "counting" algorithm
 *  for matching that is conceptually vary similar to that implemented
 *  in FwdTable.
 *
 *  \b WARNING: This implementation is provided primarily for
 *  experimental purposes.  See BTable for details.
 **/
class BCTable : public BTable {
public:
    /** @brief create and initialize a counting BTable.
     **/
    static BTable * create();
};

} // end namespace siena

#endif
